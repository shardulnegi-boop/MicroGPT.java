import java.io.*;
import java.net.URL;
import java.util.*;

/**
 * The most atomic way to train and inference a GPT in pure, dependency-free Java.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 * 
 * Ported from @karpathy's Python implementation
 */
public class MicroGPT {
    
    static Random random = new Random(42);
    
    // Hyperparameters
    static int nEmbd = 16;      // embedding dimension
    static int nHead = 4;       // number of attention heads
    static int nLayer = 1;      // number of layers
    static int blockSize = 16;  // maximum sequence length
    static int headDim = nEmbd / nHead;
    
    // Tokenizer
    static List<Character> uchars;
    static int BOS;
    static int vocabSize;
    
    // Model weights
    static Map<String, Value[][]> stateDict = new HashMap<>();
    static List<Value> params = new ArrayList<>();
    
    // Adam optimizer buffers
    static double learningRate = 0.01;
    static double beta1 = 0.85;
    static double beta2 = 0.99;
    static double epsAdam = 1e-8;
    static double[] m;
    static double[] v;
    
    // KV cache for attention
    static List<List<Value[]>> keys;
    static List<List<Value[]>> values;
    
    /**
     * Value class for autograd - tracks computation graph for backpropagation
     */
    static class Value {
        double data;
        double grad;
        Value[] children;
        double[] localGrads;
        
        Value(double data) {
            this.data = data;
            this.grad = 0;
            this.children = new Value[0];
            this.localGrads = new double[0];
        }
        
        Value(double data, Value[] children, double[] localGrads) {
            this.data = data;
            this.grad = 0;
            this.children = children;
            this.localGrads = localGrads;
        }
        
        Value add(Value other) {
            return new Value(this.data + other.data, 
                new Value[]{this, other}, 
                new double[]{1, 1});
        }
        
        Value add(double other) {
            return add(new Value(other));
        }
        
        Value mul(Value other) {
            return new Value(this.data * other.data,
                new Value[]{this, other},
                new double[]{other.data, this.data});
        }
        
        Value mul(double other) {
            return mul(new Value(other));
        }
        
        Value pow(double other) {
            return new Value(Math.pow(this.data, other),
                new Value[]{this},
                new double[]{other * Math.pow(this.data, other - 1)});
        }
        
        Value log() {
            return new Value(Math.log(this.data),
                new Value[]{this},
                new double[]{1.0 / this.data});
        }
        
        Value exp() {
            double expVal = Math.exp(this.data);
            return new Value(expVal,
                new Value[]{this},
                new double[]{expVal});
        }
        
        Value relu() {
            return new Value(Math.max(0, this.data),
                new Value[]{this},
                new double[]{this.data > 0 ? 1.0 : 0.0});
        }
        
        Value neg() {
            return this.mul(-1);
        }
        
        Value sub(Value other) {
            return this.add(other.neg());
        }
        
        Value div(Value other) {
            return this.mul(other.pow(-1));
        }
        
        void backward() {
            List<Value> topo = new ArrayList<>();
            Set<Value> visited = new HashSet<>();
            buildTopo(this, topo, visited);
            
            this.grad = 1;
            for (int i = topo.size() - 1; i >= 0; i--) {
                Value v = topo.get(i);
                for (int j = 0; j < v.children.length; j++) {
                    v.children[j].grad += v.localGrads[j] * v.grad;
                }
            }
        }
        
        private void buildTopo(Value v, List<Value> topo, Set<Value> visited) {
            if (!visited.contains(v)) {
                visited.add(v);
                for (Value child : v.children) {
                    buildTopo(child, topo, visited);
                }
                topo.add(v);
            }
        }
    }
    
    /**
     * Create a matrix of Value objects with random initialization
     */
    static Value[][] matrix(int nout, int nin, double std) {
        Value[][] mat = new Value[nout][nin];
        for (int i = 0; i < nout; i++) {
            for (int j = 0; j < nin; j++) {
                mat[i][j] = new Value(random.nextGaussian() * std);
            }
        }
        return mat;
    }
    
    static Value[][] matrix(int nout, int nin) {
        return matrix(nout, nin, 0.08);
    }
    
    /**
     * Linear layer: y = x @ W^T
     */
    static Value[] linear(Value[] x, Value[][] w) {
        Value[] result = new Value[w.length];
        for (int i = 0; i < w.length; i++) {
            Value sum = new Value(0);
            for (int j = 0; j < x.length; j++) {
                sum = sum.add(w[i][j].mul(x[j]));
            }
            result[i] = sum;
        }
        return result;
    }
    
    /**
     * Softmax activation
     */
    static Value[] softmax(Value[] logits) {
        double maxVal = Double.NEGATIVE_INFINITY;
        for (Value v : logits) {
            if (v.data > maxVal) maxVal = v.data;
        }
        
        Value[] exps = new Value[logits.length];
        Value total = new Value(0);
        for (int i = 0; i < logits.length; i++) {
            exps[i] = logits[i].sub(new Value(maxVal)).exp();
            total = total.add(exps[i]);
        }
        
        Value[] result = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            result[i] = exps[i].div(total);
        }
        return result;
    }
    
    /**
     * RMS Normalization
     */
    static Value[] rmsnorm(Value[] x) {
        Value ms = new Value(0);
        for (Value xi : x) {
            ms = ms.add(xi.mul(xi));
        }
        ms = ms.mul(1.0 / x.length);
        Value scale = ms.add(new Value(1e-5)).pow(-0.5);
        
        Value[] result = new Value[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = x[i].mul(scale);
        }
        return result;
    }
    
    /**
     * GPT forward pass
     */
    static Value[] gpt(int tokenId, int posId) {
        // Token embedding + position embedding
        Value[] tokEmb = stateDict.get("wte")[tokenId];
        Value[] posEmb = stateDict.get("wpe")[posId];
        
        Value[] x = new Value[nEmbd];
        for (int i = 0; i < nEmbd; i++) {
            x[i] = tokEmb[i].add(posEmb[i]);
        }
        x = rmsnorm(x);
        
        for (int li = 0; li < nLayer; li++) {
            // 1) Multi-head attention block
            Value[] xResidual = x;
            x = rmsnorm(x);
            
            Value[] q = linear(x, stateDict.get("layer" + li + ".attn_wq"));
            Value[] k = linear(x, stateDict.get("layer" + li + ".attn_wk"));
            Value[] vv = linear(x, stateDict.get("layer" + li + ".attn_wv"));
            
            keys.get(li).add(k);
            values.get(li).add(vv);
            
            Value[] xAttn = new Value[nEmbd];
            int attnIdx = 0;
            
            for (int h = 0; h < nHead; h++) {
                int hs = h * headDim;
                
                // Extract head slices
                Value[] qH = Arrays.copyOfRange(q, hs, hs + headDim);
                
                List<Value[]> kH = new ArrayList<>();
                List<Value[]> vH = new ArrayList<>();
                for (int t = 0; t < keys.get(li).size(); t++) {
                    kH.add(Arrays.copyOfRange(keys.get(li).get(t), hs, hs + headDim));
                    vH.add(Arrays.copyOfRange(values.get(li).get(t), hs, hs + headDim));
                }
                
                // Attention logits
                Value[] attnLogits = new Value[kH.size()];
                for (int t = 0; t < kH.size(); t++) {
                    Value sum = new Value(0);
                    for (int j = 0; j < headDim; j++) {
                        sum = sum.add(qH[j].mul(kH.get(t)[j]));
                    }
                    attnLogits[t] = sum.mul(1.0 / Math.sqrt(headDim));
                }
                
                Value[] attnWeights = softmax(attnLogits);
                
                // Weighted sum of values
                for (int j = 0; j < headDim; j++) {
                    Value headOut = new Value(0);
                    for (int t = 0; t < vH.size(); t++) {
                        headOut = headOut.add(attnWeights[t].mul(vH.get(t)[j]));
                    }
                    xAttn[attnIdx++] = headOut;
                }
            }
            
            x = linear(xAttn, stateDict.get("layer" + li + ".attn_wo"));
            for (int i = 0; i < nEmbd; i++) {
                x[i] = x[i].add(xResidual[i]);
            }
            
            // 2) MLP block
            xResidual = x;
            x = rmsnorm(x);
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc1"));
            for (int i = 0; i < x.length; i++) {
                x[i] = x[i].relu();
            }
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc2"));
            for (int i = 0; i < nEmbd; i++) {
                x[i] = x[i].add(xResidual[i]);
            }
        }
        
        return linear(x, stateDict.get("lm_head"));
    }
    
    /**
     * Reset KV cache
     */
    static void resetKVCache() {
        keys = new ArrayList<>();
        values = new ArrayList<>();
        for (int i = 0; i < nLayer; i++) {
            keys.add(new ArrayList<>());
            values.add(new ArrayList<>());
        }
    }
    
    /**
     * Sample from probability distribution
     */
    static int sample(Value[] probs) {
        double r = random.nextDouble();
        double cumsum = 0;
        for (int i = 0; i < probs.length; i++) {
            cumsum += probs[i].data;
            if (r < cumsum) return i;
        }
        return probs.length - 1;
    }
    
    public static void main(String[] args) throws Exception {
        // Load dataset
        File inputFile = new File("input.txt");
        if (!inputFile.exists()) {
            System.out.println("Downloading names dataset...");
            URL url = new URL("https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt");
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(url.openStream()));
                 PrintWriter writer = new PrintWriter(inputFile)) {
                String line;
                while ((line = reader.readLine()) != null) {
                    writer.println(line);
                }
            }
        }
        
        List<String> docs = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(inputFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) docs.add(line);
            }
        }
        Collections.shuffle(docs, random);
        System.out.println("num docs: " + docs.size());
        
        // Build tokenizer
        Set<Character> charSet = new TreeSet<>();
        for (String doc : docs) {
            for (char c : doc.toCharArray()) {
                charSet.add(c);
            }
        }
        uchars = new ArrayList<>(charSet);
        BOS = uchars.size();
        vocabSize = uchars.size() + 1;
        System.out.println("vocab size: " + vocabSize);
        
        // Initialize model weights
        stateDict.put("wte", matrix(vocabSize, nEmbd));
        stateDict.put("wpe", matrix(blockSize, nEmbd));
        stateDict.put("lm_head", matrix(vocabSize, nEmbd));
        
        for (int i = 0; i < nLayer; i++) {
            stateDict.put("layer" + i + ".attn_wq", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".attn_wk", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".attn_wv", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".attn_wo", matrix(nEmbd, nEmbd));
            stateDict.put("layer" + i + ".mlp_fc1", matrix(4 * nEmbd, nEmbd));
            stateDict.put("layer" + i + ".mlp_fc2", matrix(nEmbd, 4 * nEmbd));
        }
        
        // Flatten params
        for (Value[][] mat : stateDict.values()) {
            for (Value[] row : mat) {
                for (Value p : row) {
                    params.add(p);
                }
            }
        }
        System.out.println("num params: " + params.size());
        
        // Initialize Adam buffers
        m = new double[params.size()];
        v = new double[params.size()];
        
        // Training loop
        int numSteps = 1000;
        
        System.out.println("\n=== INITIAL WEIGHTS (before training) ===");
        System.out.print("Sample wte[0] (token 'a' embedding): [");
        for (int i = 0; i < 5; i++) {
            System.out.printf("%.4f%s", stateDict.get("wte")[0][i].data, i < 4 ? ", " : "");
        }
        System.out.println("]");
        
        for (int step = 0; step < numSteps; step++) {
            // Get document and tokenize
            String doc = docs.get(step % docs.size());
            int[] tokens = new int[doc.length() + 2];
            tokens[0] = BOS;
            for (int i = 0; i < doc.length(); i++) {
                tokens[i + 1] = uchars.indexOf(doc.charAt(i));
            }
            tokens[doc.length() + 1] = BOS;
            int n = Math.min(blockSize, tokens.length - 1);
            
            // Forward pass
            resetKVCache();
            List<Value> losses = new ArrayList<>();
            
            for (int posId = 0; posId < n; posId++) {
                int tokenId = tokens[posId];
                int targetId = tokens[posId + 1];
                
                Value[] logits = gpt(tokenId, posId);
                Value[] probs = softmax(logits);
                Value lossT = probs[targetId].log().neg();
                losses.add(lossT);
            }
            
            // Average loss
            Value loss = new Value(0);
            for (Value l : losses) {
                loss = loss.add(l);
            }
            loss = loss.mul(1.0 / n);
            
            // Backward pass
            loss.backward();
            
            // Adam optimizer update
            double lrT = learningRate * (1 - (double) step / numSteps);
            for (int i = 0; i < params.size(); i++) {
                Value p = params.get(i);
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad * p.grad;
                double mHat = m[i] / (1 - Math.pow(beta1, step + 1));
                double vHat = v[i] / (1 - Math.pow(beta2, step + 1));
                p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
                p.grad = 0;
            }
            
            System.out.printf("step %4d / %4d | loss %.4f%n", step + 1, numSteps, loss.data);
            
            // Show weight evolution at key milestones
            if (step == 0 || step == 99 || step == 999) {
                System.out.println("  === WEIGHTS AT STEP " + (step + 1) + " ===");
                System.out.print("  wte[0] (token 'a'): [");
                for (int i = 0; i < 5; i++) {
                    System.out.printf("%.4f%s", stateDict.get("wte")[0][i].data, i < 4 ? ", " : "");
                }
                System.out.println("]");
                System.out.println();
            }
        }
        
        // Inference
        double temperature = 0.5;
        System.out.println("\n--- inference (new, hallucinated names) ---");
        
        System.out.println("=== DETAILED INFERENCE STEPS ===");
        resetKVCache();
        int tokenId = BOS;
        StringBuilder sample = new StringBuilder();
        System.out.println("Starting with token: " + tokenId + " (BOS)");
        
        for (int posId = 0; posId < Math.min(5, blockSize); posId++) {
            System.out.println("\n--- Position " + posId + " ---");
            System.out.println("Input token: " + tokenId + " (" + (tokenId == BOS ? "BOS" : uchars.get(tokenId)) + ")");
            
            Value[] logits = gpt(tokenId, posId);
            System.out.print("Raw logits (first 5): [");
            for (int i = 0; i < 5; i++) {
                System.out.printf("%.3f%s", logits[i].data, i < 4 ? ", " : "");
            }
            System.out.println("]");
            
            // Apply temperature
            Value[] scaledLogits = new Value[logits.length];
            for (int i = 0; i < logits.length; i++) {
                scaledLogits[i] = logits[i].mul(1.0 / temperature);
            }
            Value[] probs = softmax(scaledLogits);
            
            System.out.print("Probabilities (first 5): [");
            for (int i = 0; i < 5; i++) {
                System.out.printf("%.3f%s", probs[i].data, i < 4 ? ", " : "");
            }
            System.out.println("]");
            
            tokenId = sample(probs);
            System.out.println("Selected token: " + tokenId + " (" + (tokenId == BOS ? "BOS" : uchars.get(tokenId)) + ")");
            
            if (tokenId == BOS) {
                System.out.println("Got BOS token, stopping");
                break;
            }
            sample.append(uchars.get(tokenId));
            System.out.println("Current sample: '" + sample + "'");
        }
        
        System.out.println("\nFinal generated name: '" + sample + "'");
        System.out.println("\n=== REST OF SAMPLES (fast mode) ===");
        
        // Generate remaining samples
        for (int sampleIdx = 1; sampleIdx < 20; sampleIdx++) {
            resetKVCache();
            tokenId = BOS;
            sample = new StringBuilder();
            
            for (int posId = 0; posId < blockSize; posId++) {
                Value[] logits = gpt(tokenId, posId);
                Value[] scaledLogits = new Value[logits.length];
                for (int i = 0; i < logits.length; i++) {
                    scaledLogits[i] = logits[i].mul(1.0 / temperature);
                }
                Value[] probs = softmax(scaledLogits);
                tokenId = sample(probs);
                
                if (tokenId == BOS) break;
                sample.append(uchars.get(tokenId));
            }
            
            System.out.println("sample " + String.format("%2d", sampleIdx + 1) + ": " + sample);
        }
    }
}

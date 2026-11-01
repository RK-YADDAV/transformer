import math
from random import random

class Tensor:
    def __init__(self, data, shape=None):
        if shape is not None:
            self.shape = shape
        else:
            self.shape = self._get_shape(data)

        self.data = self._flatten(data)

    def _get_shape(self, data):
        if isinstance(data, list) and len(data) == 0:
            return (0,)

        if isinstance(data, list):
            return (len(data),) + self._get_shape(data[0])

        return ()

    def _flatten(self, data):
        if not isinstance(data, list):
            return [data if isinstance(data, Value) else Value(data)]
    
        result = []
    
        for item in data:
            result.extend(self._flatten(item))
    
        return result
        
    def __mul__(self, scalar):
        result = []

        for value in self.data:
            result.append(value * scalar)

        return Tensor(result, shape=self.shape)

    def __truediv__(self, scalar):
        result = []

        for value in self.data:
            result.append(value / scalar)

        return Tensor(result, shape=self.shape)

    def __add__(self, other):
        # Case 1: Same shape
        if self.shape == other.shape:
            result = []

            for a, b in zip(self.data, other.data):
                result.append(a + b)

            return Tensor(result, shape=self.shape)

        # Case 2: Broadcasting
        if (
            len(self.shape) == 2
            and len(other.shape) == 2
            and other.shape[0] == 1
            and self.shape[1] == other.shape[1]
        ):
            cols = self.shape[1]
            result = []

            for k, value in enumerate(self.data):
                result.append(value + other.data[k % cols])

            return Tensor(result, shape=self.shape)

        # Fallback
        raise ValueError(
            f"Cannot add tensors with shapes {self.shape} and {other.shape}"
        )
    def __matmul__(self, other):
        # Shape validation
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")
    
        rows_a, cols_a = self.shape
        rows_b, cols_b = other.shape
    
        if cols_a != rows_b:
            raise ValueError(
                f"Cannot multiply shapes {self.shape} and {other.shape}"
            )
    
        # Setup
        result = []
    
        # Rows of A
        for i in range(rows_a):
    
            # Columns of B
            for j in range(cols_b):
    
                dot_product = 0.0
    
                # Dot product
                for k in range(cols_a):
                    index_a = i * cols_a + k
                    index_b = k * cols_b + j
    
                    dot_product += self.data[index_a] * other.data[index_b]
    
                result.append(dot_product)
    
        return Tensor(result, shape=(rows_a, cols_b))
        
    def transpose(self):
        # Ensure tensor is 2D
        if len(self.shape) != 2:
            raise ValueError("Transpose requires a 2D tensor")
    
        rows, cols = self.shape
        result = []
    
        # Column-first traversal
        for j in range(cols):
            for i in range(rows):
                index = i * cols + j
                result.append(self.data[index])
    
        return Tensor(result, shape=(cols, rows))

    def cat(self, other):
        # Check row counts
        if self.shape[0] != other.shape[0]:
            raise ValueError("Cannot concatenate tensors with different row counts")
    
        rows = self.shape[0]
        cols_a = self.shape[1]
        cols_b = other.shape[1]
    
        result = []
    
        for i in range(rows):
            row_a = self.data[i * cols_a : (i + 1) * cols_a]
            row_b = other.data[i * cols_b : (i + 1) * cols_b]
    
            result.extend(row_a + row_b)
    
        return Tensor(result, shape=(rows, cols_a + cols_b))
    
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"
        
        
class Module:
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Tensor([
            [random.gauss(0.0, 0.02) for _ in range(out_features)]
            for _ in range(in_features)
        ])

        self.bias = Tensor([
            [0.0 for _ in range(out_features)]
        ])

    def forward(self, x):
        return (x @ self.weight) + self.bias



class Embedding(Module):
    def __init__(self, vocab_size, embedding_dim):
        self.weight = Tensor([
            [random.gauss(0.0, 0.02) for _ in range(embedding_dim)]
            for i in range(vocab_size)
        ])

    def forward(self, x):
        embed_dim = self.weight.shape[1]
        result = []

        for token_id in x:
            start = token_id * embed_dim
            end = start + embed_dim

            result.extend(self.weight.data[start:end])

        return Tensor(result, shape=(len(x), embed_dim))
        


class ReLU(Module):
    def forward(self, x):
        result = []

        for value in x.data:
            result.append(value.relu())

        return Tensor(result, shape=x.shape)


class Softmax(Module):
    def forward(self, x):
        rows, cols = x.shape
        result = []
    
        for i in range(rows):
            start = i * cols
            row = x.data[start:start + cols]
    
            row_max = max(v.data for v in row)
    
            exp_values = [
                (v - row_max).exp()
                for v in row
            ]
    
            exp_sum = sum(exp_values)
    
            normalized = [
                value / exp_sum
                for value in exp_values
            ]
    
            result.extend(normalized)
    
        return Tensor(result, shape=x.shape)


class LayerNorm(Module):
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def forward(self, x):
        rows, cols = x.shape
        result = []

        for i in range(rows):
            start = i * cols
            row = x.data[start:start + cols]

            mean = sum(row) / cols

            variance = sum(
                (value - mean) ** 2
                for value in row
            ) / cols

            normalized = [
                (value - mean) / ((variance + 1e-5) ** 0.5)
                for value in row
            ]

            result.extend(normalized)

        return Tensor(result, shape=x.shape)
        

class SelfAttention(Module):
    def __init__(self, embed_dim, head_dim):
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.q_proj = Linear(embed_dim, head_dim)
        self.k_proj = Linear(embed_dim, head_dim)
        self.v_proj = Linear(embed_dim, head_dim)

        self.softmax = Softmax()

    def forward(self, x):
        # Generate Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Calculate attention scores
        scores = Q @ K.transpose()
        scores = scores / math.sqrt(self.head_dim)

        # Causal mask
        rows, cols = scores.shape

        for i in range(rows):
            for j in range(cols):
                if j > i:
                    idx = i * cols + j
                    scores.data[idx] = Value(-float('inf'))

        # Convert scores to attention weights
        weights = self.softmax(scores)

        # Weighted sum of V
        return weights @ V


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        self.heads = []

        for _ in range(num_heads):
            self.heads.append(
                SelfAttention(embed_dim, head_dim)
            )

        self.o_proj = Linear(
            num_heads * head_dim,
            embed_dim
        )

    def forward(self, x):
        head_outputs = []

        # Run input through every attention head
        for head in self.heads:
            head_outputs.append(head(x))

        # Concatenate outputs
        combined = head_outputs[0]

        for i in range(1, len(head_outputs)):
            combined = combined.cat(head_outputs[i])

        # Final output projection
        return self.o_proj(combined)
        
        
class FeedForward(Module):
    def __init__(self, embed_dim):
        self.fc1 = Linear(embed_dim, 4 * embed_dim)
        self.relu = ReLU()
        self.fc2 = Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim,
            num_heads,
            head_dim
        )
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        # Step 1: Pre-LN → Attention → Residual
        x = x + self.attn(self.ln1(x))

        # Step 2: Pre-LN → FeedForward → Residual
        x = x + self.ffn(self.ln2(x))

        return x
        
class Transformer(Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        head_dim,
        num_layers
    ):
        # Token and positional embeddings
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.blocks = []

        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    head_dim
                )
            )

        # Final normalization and language-model head
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size)

    def forward(self, x):
        # Token embeddings
        token_embeddings = self.token_emb(x)

        # Position embeddings
        positions = list(range(len(x)))
        position_embeddings = self.pos_emb(positions)

        # Combine token + position information
        h = token_embeddings + position_embeddings

        # Pass through Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Final normalization
        h = self.ln_f(h)

        # Convert hidden states to vocabulary logits
        logits = self.lm_head(h)

        return logits
        
        
class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # Reverse multiplication
    def __rmul__(self, other):
        return self * other

    # Reverse addition
    def __radd__(self, other):
        return self + other

    # Power
    def __pow__(self, other):
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # Division
    def __truediv__(self, other):
        return self * (other ** -1)

    # Negation
    def __neg__(self):
        return self * -1

    # Subtraction
    def __sub__(self, other):
        return self + (-other)

    # Reverse subtraction
    def __rsub__(self, other):
        return other + (-self)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
        
    def relu(self):
        out = Value(
            0 if self.data < 0 else self.data,
            (self,)
        )

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(
            math.exp(self.data),
            (self,)
        )

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
        
    def log(self):
        out = Value(math.log(self.data), (self,))

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward

        return out

        
        
def get_parameters(obj):
    params = []

    if isinstance(obj, Tensor):
        params.extend(obj.data)

    elif hasattr(obj, '__dict__'):
        for key, val in obj.__dict__.items():
            if isinstance(val, list):
                for item in val:
                    params.extend(get_parameters(item))
            else:
                params.extend(get_parameters(val))

    return params
        
        
def generate(model, input_tokens, max_new_tokens):
    for _ in range(max_new_tokens):
        # Get model predictions
        logits = model(input_tokens)

        # We only care about the predictions for the very last token in the sequence
        # Because logits is 1D, we slice out the last row:
        vocab_size = logits.shape[1]
        start_idx = (len(input_tokens) - 1) * vocab_size
        last_row = logits.data[start_idx : start_idx + vocab_size]

        # Find the index of the highest score (argmax)
        # Bridge out of the Autograd engine by extracting the raw floats
        last_row_floats = [v.data for v in last_row]
        next_token = last_row_floats.index(max(last_row_floats))

        # Append generated token to the sequence
        input_tokens.append(next_token)

    return input_tokens
    
def measure_latency(model, seq_len, runs=10):
    # Create a sequence of the required length
    input_tokens = [i % 50 for i in range(seq_len)]

    # Warm-up runs
    for _ in range(2):
        model(input_tokens)

    times = []

    for _ in range(runs):
        start = time.perf_counter()

        model(input_tokens)

        end = time.perf_counter()

        times.append((end - start) * 1000)  # milliseconds

    average_latency = sum(times) / len(times)

    return average_latency
    
    
def test_future_token_gradient_leakage():
    print("\nFuture-Token Gradient Leakage Test")
    print("=" * 50)

    # Use a small attention module
    embed_dim = 4
    head_dim = 4
    seq_len = 4

    attn = SelfAttention(embed_dim, head_dim)

    # ------------------------------------------------------------
    # Give the attention weights non-identical values.
    # This makes the gradient test meaningful.
    # ------------------------------------------------------------
    params = get_parameters(attn)

    for i, p in enumerate(params):
        p.data = 0.01 * (i + 1)

    # Deliberately non-uniform input
    input_data = [
        [0.2, -0.1, 0.3, 0.7],
        [0.5,  0.4, -0.2, 0.1],
        [-0.3, 0.8, 0.6, -0.5],
        [0.9, -0.7, 0.2, 0.4]
    ]

    maximum_future_gradient = 0.0

    # Test every output position
    for i in range(seq_len):

        # Fresh input graph for this test
        x = Tensor(input_data)

        # Forward
        output = attn(x)

        # Select ONE scalar from output position i
        idx = i * head_dim

        loss = output.data[idx]

        # Backward
        loss.backward()

        # Check gradients of future input positions
        future_gradients = []

        for j in range(i + 1, seq_len):

            start = j * embed_dim
            end = start + embed_dim

            grads = [
                abs(v.grad)
                for v in x.data[start:end]
            ]

            future_gradients.extend(grads)

        if future_gradients:
            future_max = max(future_gradients)
        else:
            future_max = 0.0

        maximum_future_gradient = max(
            maximum_future_gradient,
            future_max
        )

        print(
            f"Output position {i} | "
            f"Maximum future-token gradient: {future_max:.12f}"
        )

    print("=" * 50)

    print(
        f"Maximum gradient leakage observed: "
        f"{maximum_future_gradient:.12f}"
    )

    if maximum_future_gradient == 0.0:
        print("PASS: Future-token gradient leakage = exactly 0.0")
    else:
        print("FAIL: Future-token gradient leakage detected!")
    
    
    
    
    
if __name__ == "__main__":

    # ============================================================
    # CONFIGURATION
    # ============================================================

    vocab_size = 50
    max_seq_len = 10
    learning_rate = 0.01

    # ============================================================
    # ASK FOR INPUT BEFORE TRAINING
    # ============================================================

    print("=" * 50)
    print("TRANSFORMER ENGINE")
    print("=" * 50)

    try:
        raw_input_seq = input(
            "Enter input token IDs separated by commas "
            "(e.g. 5,11,15): "
        )

        parts = raw_input_seq.split(",")

        if not raw_input_seq.strip() or any(
            not part.strip() for part in parts
        ):
            raise ValueError("Input sequence cannot be empty.")

        input_seq = [int(part.strip()) for part in parts]

        if any(token < 0 or token >= vocab_size for token in input_seq):
            raise ValueError("Token IDs must be between 0 and 49.")

        if len(input_seq) >= max_seq_len:
            raise ValueError(
                "Input sequence must have fewer than 10 tokens "
                "to generate at least one new token."
            )

        target_token = int(
            input(
                "Enter the expected next token ID (0-49): "
            )
        )

        if not 0 <= target_token < vocab_size:
            raise ValueError(
                "Target token must be between 0 and 49."
            )

        epochs = int(
            input("Enter number of training iterations: ")
        )

        if epochs < 1:
            raise ValueError(
                "Training iterations must be at least 1."
            )

    except ValueError as error:
        print("Invalid input:", error)
        raise SystemExit(1)

    # ============================================================
    # CREATE MODEL
    # ============================================================

    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=16,
        num_heads=2,
        head_dim=8,
        num_layers=2
    )

    params = get_parameters(model)

    # ============================================================
    # TRAIN ON THE USER-PROVIDED EXAMPLE
    # ============================================================

    print("\nTraining started...")
    print("Training input:", input_seq)
    print("Expected next token:", target_token)

    for epoch in range(epochs):

        # Forward pass
        logits = model(input_seq)

        # Extract logits from the final input position
        rows, cols = logits.shape
        start = (rows - 1) * cols
        last_row = logits.data[start:start + cols]

        # Convert final-position logits into probabilities
        probs = Softmax().forward(
            Tensor([last_row])
        ).data

        # Calculate loss for the user's target token
        p_target = probs[target_token]
        # Add 1e-8 to prevent math.log(0.0) crashes
        loss = -(p_target + 1e-8).log()

        # Save the loss value before updating parameters
        loss_value = loss.data

        # Backward pass
        loss.backward()

        # Gradient-descent parameter update
        for p in params:
            p.data -= learning_rate * p.grad

        # Reset gradients for the next iteration
        for p in params:
            p.grad = 0.0

        print(
            f"Iteration {epoch + 1}/{epochs} "
            f"| Loss: {loss_value:.8f}"
        )

    # ============================================================
    # GENERATE AFTER TRAINING
    # ============================================================

    print("\nTraining complete! Testing the model...")

    # Preserve the user's original input
    original_input = input_seq.copy()

    # Generate one new token
    final_output = generate(
        model,
        original_input.copy(),
        max_new_tokens=1
    )

    # Extract only the newly predicted token(s)
    predicted_tokens = final_output[len(original_input):]

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)

    print("Input sequence:          ", original_input)
    print("Expected next token:     ", target_token)
    print("Model predicted token:   ", predicted_tokens[0])
    print("Complete output sequence:", final_output)

    print("=" * 50)

    # ============================================================
    # LATENCY BENCHMARK
    # ============================================================

    print("\nLatency Benchmark")

    for seq_len in [1, 2, 4, 8]:

        latency = measure_latency(
            model,
            seq_len,
            runs=10
        )

        print(
            f"Sequence Length: {seq_len:2d} | "
            f"Average Latency: {latency:.2f} ms"
        )


    # ============================================================
    # CALCULATE 1 -> 8 RATIO
    # ============================================================

    latency_1 = measure_latency(
        model,
        seq_len=1,
        runs=10
    )

    latency_8 = measure_latency(
        model,
        seq_len=8,
        runs=10
    )

    ratio = latency_8 / latency_1

    print(
        f"\nLatency ratio (1 → 8 tokens): {ratio:.2f}x"
    )
    
    
    
test_future_token_gradient_leakage()
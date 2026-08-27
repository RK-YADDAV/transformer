import math

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
            [0.01 for _ in range(out_features)]
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
            [(i + 1) * 0.01 for _ in range(embedding_dim)]
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
    
if __name__ == "__main__":
    # Create model
    model = Transformer(
        vocab_size=50,
        max_seq_len=10,
        embed_dim=16,
        num_heads=2,
        head_dim=8,
        num_layers=2
    )

    learning_rate = 0.01

    params = get_parameters(model)

    # Simple dataset
    input_seq = [5, 10, 15]
    target_token = 20

    # Training
    for epoch in range(10):

        # Forward
        logits = model(input_seq)

        # Extract last row
        rows, cols = logits.shape
        start = (rows - 1) * cols
        last_row = logits.data[start:start + cols]

        # Probabilities
        probs = Softmax().forward(
            Tensor([last_row])  # <-- Add brackets here to make it 2D!
        ).data

        # Loss
        p_target = probs[target_token]
        loss = -p_target.log()

        # Backward
        loss.backward()

        # Update parameters
        for p in params:
            p.data -= learning_rate * p.grad

        # Zero gradients
        for p in params:
            p.grad = 0.0

        # Print
        print(f"Epoch {epoch} | Loss: {loss.data:.4f}")
        
        
# --- Add this at the very bottom, outside the training loop ---
    print("\nTraining complete! Testing the model...")
    
    # Give it the starting sequence
    test_seq = [5, 10, 15]
    
    # Ask it to generate 1 new token using its newly trained weights
    final_output = generate(model, test_seq, max_new_tokens=1)
    
    print(f"Generated sequence: {final_output}")
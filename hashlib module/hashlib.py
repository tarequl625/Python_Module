# ================================
# hashlib module in Python
# ================================
# hashlib is used for hashing (generating fixed-size digests)
# Supports binary, hexadecimal, and other formats

import hashlib

# -----------------------------
# Show available algorithms
# -----------------------------
print("Guaranteed algorithms:", hashlib.algorithms_guaranteed)
print("Available algorithms:", hashlib.algorithms_available)
# Example output:
# Guaranteed algorithms: {'sha512', 'sha384', 'sha224', 'sha1', 'md5', 'sha256'}
# Available algorithms: {'sha512', 'sha3_512', 'sha384', 'blake2b', 'sha3_256', 'sha1', ...}

# -----------------------------
# TYPE 1: Direct hashing with message
# -----------------------------
message = b"Hello World"  # Must be bytes

# MD5 (128-bit)
print(hashlib.md5(message).digest())      # Binary output
print(hashlib.md5(message).hexdigest())   # Hex output
# Example hexdigest: e59ff97941044f85df5297e1c302d260

# SHA1 (160-bit)
print(hashlib.sha1(message).digest())
print(hashlib.sha1(message).hexdigest())
# Example hexdigest: 2ef7bde608ce5404e97d5f042f95f89f1c232871

# SHA224 (224-bit)
print(hashlib.sha224(message).digest())
print(hashlib.sha224(message).hexdigest())
# Example hexdigest: 730e109bd7a8a32b1cb9d9a0e9a5f5f7ffcf64f29d0f6c3c2bf7a6d2

# SHA256 (256-bit)
print(hashlib.sha256(message).digest())
print(hashlib.sha256(message).hexdigest())
# Example hexdigest: a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b5e52b98b

# SHA384 (384-bit)
print(hashlib.sha384(message).digest())
print(hashlib.sha384(message).hexdigest())

# SHA512 (512-bit)
print(hashlib.sha512(message).digest())
print(hashlib.sha512(message).hexdigest())

# SHA3-256
print(hashlib.sha3_256(message).digest())
print(hashlib.sha3_256(message).hexdigest())

# SHA3-384
print(hashlib.sha3_384(message).digest())
print(hashlib.sha3_384(message).hexdigest())

# SHA3-512
print(hashlib.sha3_512(message).digest())
print(hashlib.sha3_512(message).hexdigest())

# SHAKE-128 (16 bytes)
print(hashlib.shake_128(message).digest(16))
print(hashlib.shake_128(message).hexdigest(16))

# SHAKE-256 (16 bytes)
print(hashlib.shake_256(message).digest(16))
print(hashlib.shake_256(message).hexdigest(16))

# BLAKE2b
print(hashlib.blake2b(message).digest())
print(hashlib.blake2b(message).hexdigest())

# BLAKE2s
print(hashlib.blake2s(message).digest())
print(hashlib.blake2s(message).hexdigest())

# -----------------------------
# TYPE 2: Using update() method
# -----------------------------
m = hashlib.sha1()  # create SHA1 object
m.update(message)   # feed data (can be multiple chunks)
print(m.digest())   # Binary digest
# Same hexdigest as above SHA1

# -----------------------------
# TYPE 3: Using hashlib.new()
# -----------------------------
n = hashlib.new('sha1')
n.update(message)
print(n.digest())
print(n.digest_size)  # e.g., 20 bytes
print(n.block_size)   # e.g., 64 bytes
print(n.name)         # e.g., 'sha1'
print(n.copy)         # method reference

# -----------------------------
# TYPE 4: Hashing a file
# -----------------------------
with open("D:\\pd.pdf", "rb") as f:
    digest = hashlib.file_digest(f, "sha256")  # SHA256 hash of file
print(digest.hexdigest())
# Example output (hex string): d2d2ae1f0c13c52f...

# -----------------------------
# TYPE 5: PBKDF2 Key Derivation
# -----------------------------
a = hashlib.pbkdf2_hmac("sha256", b"abc", b"salt", 2000)
print(a.hex())
# Example output: 9b8c4f7b6e1a2f5d4c8b9f1a2d3e4f5c6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1
# ================================
# hashlib module in Python
# ================================
# hashlib is used for hashing (generating fixed-size digests).
# Supports binary, hexadecimal, and other formats.

import hashlib

# Show available algorithms
print(hashlib.algorithms_guaranteed)   # Algorithms guaranteed to work in all platforms
print(hashlib.algorithms_available)    # Algorithms available in the current system

# --------------------------------
# TYPE 1: Direct Hashing with message
# --------------------------------
message = b"Hello World"  # Input must be in bytes, not string

# MD5 (128-bit) → Fast but broken, used only for simple checksums
print(hashlib.md5(message).digest())
print(hashlib.md5(message).hexdigest())

# SHA1 (160-bit) → Stronger than MD5, but now considered insecure
print(hashlib.sha1(message).digest())
print(hashlib.sha1(message).hexdigest())

# SHA224 (224-bit) → Truncated version of SHA256
print(hashlib.sha224(message).digest())
print(hashlib.sha224(message).hexdigest())

# SHA256 (256-bit) → Very common, secure, used in Bitcoin, SSL, etc.
print(hashlib.sha256(message).digest())
print(hashlib.sha256(message).hexdigest())

# SHA384 (384-bit) → Secure, longer digest, often for digital signatures
print(hashlib.sha384(message).digest())
print(hashlib.sha384(message).hexdigest())

# SHA512 (512-bit) → Very strong, widely used in modern cryptography
print(hashlib.sha512(message).digest())
print(hashlib.sha512(message).hexdigest())

# SHA3-256 (256-bit) → From SHA3 family, based on Keccak algorithm
print(hashlib.sha3_256(message).digest())
print(hashlib.sha3_256(message).hexdigest())

# SHA3-384 (384-bit) → SHA3 family version
print(hashlib.sha3_384(message).digest())
print(hashlib.sha3_384(message).hexdigest())

# SHA3-512 (512-bit) → Strong SHA3 variant
print(hashlib.sha3_512(message).digest())
print(hashlib.sha3_512(message).hexdigest())

# SHAKE-128 (XOF: extendable output function) → Digest length is flexible
print(hashlib.shake_128(message).digest(16))
print(hashlib.shake_128(message).hexdigest(16))

# SHAKE-256 (XOF, stronger than SHAKE-128, flexible output size)
print(hashlib.shake_256(message).digest(16))
print(hashlib.shake_256(message).hexdigest(16))

# BLAKE2b (up to 512-bit digest) → Very fast and secure, modern replacement
print(hashlib.blake2b(message).digest())
print(hashlib.blake2b(message).hexdigest())

# BLAKE2s (up to 256-bit digest) → Lightweight, faster for smaller data
print(hashlib.blake2s(message).digest())
print(hashlib.blake2s(message).hexdigest())

# --------------------------------
# TYPE 2: Using update() method
# --------------------------------
m = hashlib.sha1()          # Create SHA1 object
m.update(message)           # Feed data in chunks (can call multiple times)
print(m.digest())           # Binary digest

# --------------------------------
# TYPE 3: Using hashlib.new()
# --------------------------------
n = hashlib.new('sha1')     # Create hash object by algorithm name
n.update(message)
print(n.digest())           # Binary digest
print(n.digest_size)        # Digest size (in bytes)
print(n.block_size)         # Internal block size
print(n.name)               # Algorithm name
print(n.copy)               # Method reference for copying hash object

# --------------------------------
# TYPE 4: Hashing a file
# --------------------------------
with open("D:\\pd.pdf", "rb") as f:                     # Open file in binary mode
    digest = hashlib.file_digest(f, "sha256")           # Compute SHA256 hash of file
print(digest.hexdigest())                               # Print hex digest

# --------------------------------
# TYPE 5: Key Derivation with PBKDF2
# --------------------------------
# pbkdf2_hmac → Password-based key derivation function
# Parameters: (hash function, password, salt, iterations)
a = hashlib.pbkdf2_hmac("sha256", b"abc", b"salt", 2000)
print(a.hex())  # Convert derived key (binary) to hex string
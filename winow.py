import hashlib

def hash_function(input_string):
    return int(hashlib.sha256(input_string.encode()).hexdigest(), 16)

def generate_k_grams(text, k):
    return [text[i:i + k] for i in range(len(text) - k + 1)]

def winnowing(text, k, window_size):

    if not text or len(text) < k:
        return [] 
    
    k_grams = generate_k_grams(text, k)
    hashes = [hash_function(k_gram) for k_gram in k_grams]
    fingerprints = []
    min_hash_index = -1
    for i in range(len(hashes) - window_size + 1):
        window = hashes[i:i + window_size]
        min_hash = min(window)     
        if min_hash_index == -1 or min_hash != hashes[min_hash_index]:
            min_hash_index = i + window.index(min_hash)
            fingerprints.append((min_hash, min_hash_index))

    return fingerprints

def compare_texts(text1, text2, k=5, window_size=4):
    if not text1 or not text2:
        return 0.0, set()  
    fingerprints1 = winnowing(text1, k, window_size)
    fingerprints2 = winnowing(text2, k, window_size)

    if not fingerprints1 or not fingerprints2:
        return 0.0, set()  

    
    hash_set1 = set([fp[0] for fp in fingerprints1])
    hash_set2 = set([fp[0] for fp in fingerprints2])

    common_fingerprints = hash_set1.intersection(hash_set2)

    if not common_fingerprints:
        return 0.0, set()  

    similarity = len(common_fingerprints) / min(len(hash_set1), len(hash_set2))
    return similarity




def similarity_score(text1,text2):
    similarity = compare_texts(text1, text2)
    return similarity*100

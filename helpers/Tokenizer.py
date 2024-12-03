import regex as re

class TokenProcessor:
    def __init__(self, filepath, vocab_size=276):
        self.filepath = filepath
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def _load_text(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.tokens = list(map(int, self.text.encode('utf-8')))
        print("Text length:", len(self.text))
        print("Token length:", len(self.tokens))

    def _process_text(self):
        ids = list(self.tokens)
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx

        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.tokenized_data = ids

    def _get_stats(self, ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def find_tokens(self, pattern, text):
        compiled_pattern = re.compile(pattern)
        return re.findall(compiled_pattern, text)

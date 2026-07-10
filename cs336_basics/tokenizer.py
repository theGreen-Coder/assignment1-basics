import regex as re
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
import json
from cs336_basics.pretokenization_example import find_chunk_boundaries

def count_pre_tokenize(file_path:str, start:int, end:int, regex_pattern:str, special_tokens:list[str], q:Queue):
	process_dict = {}
	with open(file_path, "rb") as f:
		f.seek(start)
		chunk = f.read(end - start).decode("utf-8", errors="ignore")

		# Split on special tokens
		split_special_tokens = re.split("|".join([re.escape(tok) for tok in special_tokens]), chunk)

		for split in tqdm(split_special_tokens):
			# Split text into pretokens using GPT2 regex
			pre_tokens = re.findall(regex_pattern, split)

			for tok in pre_tokens:
				process_dict[tok] = process_dict.get(tok, 0) + 1

	process_dict_binary = {}
	for w, freq in process_dict.items():
		process_dict_binary[tuple(w.encode("utf-8"))] = freq
	
	q.put(process_dict_binary)

def merge_dictionaries(dict_list):
	merged = defaultdict(int)
	for d in dict_list:
		for key, value in d.items():
			merged[key] += value
	return dict(merged)

def get_all_pairs(t:tuple|list):
	return [(tok1, tok2) for tok1, tok2 in zip(t[:-1], t[1:])]

def find_max(pair_counts, vocab):
	max_freq = -1
	max_pair = (-1, -1)
	for pair, freq in pair_counts.items():
		if freq > max_freq:
			max_freq = freq
			max_pair = pair
		if freq == max_freq:
			if vocab[pair[0]] > vocab[max_pair[0]]:
				max_pair = pair
			elif vocab[pair[0]] == vocab[max_pair[0]]:
				if vocab[pair[1]] > vocab[max_pair[1]]:
					max_pair = pair
	return max_pair

def train_bpe_tokenizer(input_path:str, vocab_size:int, special_tokens:list[str], num_processes=1, verbose=True) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
	# Pattern for pre-tokenization
	PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

	# Pre-tokenization (parallelized)
	proceses = []
	dictionaries = Queue()
	dictionaries_list = []
	
	# starting pre-tokenization
	with open(input_path, "rb") as f:
		boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

		# A bit of multiprocessing magic to make things faster
		for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
			p = Process(target=count_pre_tokenize, args=(input_path, start, end, PAT, special_tokens, dictionaries))
			p.start()
			proceses.append(p)

		for _ in range(num_processes):
			dictionaries_list.append(dictionaries.get())

		for p in proceses:
			p.join()
	
	# Merging the dictionaries from all the processes
	pre_tokens = merge_dictionaries(dictionaries_list)
	pair_counts = {}
	pair_to_pre_token = {}
	
	# Count pair frequencies in pre-tokens at initialization
	for pre_tok_idx, count in pre_tokens.items():
		for pair in get_all_pairs(pre_tok_idx):
			pair_counts[pair] = pair_counts.get(pair, 0) + count

			if pair not in pair_to_pre_token:
				pair_to_pre_token[pair] = set()
			pair_to_pre_token[pair].add(tuple(pre_tok_idx))
	
	vocab = {idx:bytes([idx]) for idx in range(256)}

	# add sepcial tokens to vocab
	for tok in special_tokens:
		vocab[len(vocab)] = tok.encode("utf-8")

	# BPE merges
	if verbose: print("starting BPE merges!")
	merges = []
	for _ in tqdm(range(vocab_size - len(vocab))):
		max_pair = find_max(pair_counts, vocab)

		# Update vocab and merges
		token_bytes1 = vocab[max_pair[0]]
		token_bytes2 = vocab[max_pair[1]]
		merges.append((token_bytes1, token_bytes2))
		new_idx_token = len(vocab)
		vocab[new_idx_token] = b''.join([token_bytes1, token_bytes2])

		# Update to corrected pair counts
		for old_pre_tok in pair_to_pre_token[max_pair]:
			old_pre_tok_freq = pre_tokens[old_pre_tok]
			new_pre_tok = []

			# updating pair_counts and pair_to_pre_token
			for pair in get_all_pairs(old_pre_tok):
				pair_counts[pair] -= old_pre_tok_freq
				if pair != max_pair and old_pre_tok in pair_to_pre_token[pair]:
					pair_to_pre_token[pair].remove(old_pre_tok)

			# getting the new token sequence with the new token in it
			idx = 0 
			while idx < len(old_pre_tok):
				if idx == len(old_pre_tok)-1:
					new_pre_tok.append(old_pre_tok[idx])
					idx += 1
				else:
					if (old_pre_tok[idx], old_pre_tok[idx+1]) == max_pair:
						new_pre_tok.append(new_idx_token)
						idx += 2
						continue
					else:
						new_pre_tok.append(old_pre_tok[idx])
						idx += 1
			
			# setting the new merged token's frequency
			new_pre_tok = tuple(new_pre_tok)
			pre_tokens[new_pre_tok] = old_pre_tok_freq
			del pre_tokens[old_pre_tok]

			# update pair_to_pre_token
			for pair in get_all_pairs(new_pre_tok):
				pair_counts[pair] = pair_counts.get(pair, 0) + old_pre_tok_freq

				if pair not in pair_to_pre_token:
					pair_to_pre_token[pair] = set()
				else:
					if old_pre_tok in pair_to_pre_token[pair]:
						pair_to_pre_token[pair].remove(old_pre_tok)
				
				pair_to_pre_token[pair].add(new_pre_tok)
		
		del pair_to_pre_token[max_pair]
		del pair_counts[max_pair]

	return vocab, merges

class Tokenizer:
	def __init__(self, vocab:dict[int, bytes], bpe_merges:list[tuple[bytes,bytes]], special_tokens=None, verbose=True):
		if verbose: print("Initializing Tokenizer!")
		self.vocab = vocab
		self.i_vocab = {v:k for k, v in vocab.items()}
		self.special_tokens = set() if special_tokens is None else sorted(set(special_tokens), key=len, reverse=True)
		self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
		self.verbose = verbose

		self.bpe_merges = {(self.i_vocab[b1], self.i_vocab[b2]):self.i_vocab[b1 + b2] for (b1, b2) in bpe_merges}
		self.bpe_merges_ranks = {(self.i_vocab[tu[0]], self.i_vocab[tu[1]]):idx for idx, tu in enumerate(bpe_merges)}

		self.b_int_to_token = {idx:self.i_vocab[bytes([idx])] for idx in range(256)}
		self.pre_token_cache = {}
		self.pre_token_utf8_cache = {}
	
	def _to_utf8(self, pre_tok):
		if pre_tok in self.special_tokens:
			return (self.i_vocab[pre_tok.encode("utf-8")],)
		elif pre_tok in self.pre_token_utf8_cache:
			return self.pre_token_utf8_cache[pre_tok]
		else:
			tmp = []
			for b in pre_tok.encode("utf-8"):
				tmp.append(self.b_int_to_token[b])
			self.pre_token_utf8_cache[pre_tok] = tuple(tmp)
			return tuple(tmp)

	def encode(self, text:str) -> list[int]:
		if self.special_tokens is not None and len(self.special_tokens) > 0:
			text = re.split("(" + "|".join([re.escape(tok) for tok in self.special_tokens]) + ")", text)
		else:
			text = [text]

		if self.verbose: print("pre_tokenization")
		pre_tokens = []
		for split in tqdm(text):
			if split in self.special_tokens:
				pre_tokens.append(split)
			else:
				pre_tokens.extend(re.findall(self.PAT, split))
		
		if self.verbose: print("starting encoding!")

		output_tokens = []
		for pre_tok in tqdm(pre_tokens):
			tokens = self._to_utf8(pre_tok)
			assert type(tokens) == tuple
			if tokens in self.pre_token_cache:
				output_tokens.extend(self.pre_token_cache[tokens])
			else:
				old_tokens = list(tokens)
				
				while len(old_tokens) > 1:
					# find maximum ranked bpe pair
					max_rank = float('inf')
					max_pair = -1
					for pair in get_all_pairs(old_tokens):
						if pair in self.bpe_merges_ranks:
							rank = self.bpe_merges_ranks[pair]
							if rank < max_rank:
								max_rank = rank
								max_pair = pair

					if max_rank == float('inf'):
						break
					
					merge = self.bpe_merges[max_pair]
					
					# substitute the bpe merge
					idx = 0
					new_tokens = []
					while idx < len(old_tokens):
						if idx == len(old_tokens)-1:
							new_tokens.append(old_tokens[idx])
							idx += 1
						else:
							if (old_tokens[idx], old_tokens[idx+1]) == max_pair:
								new_tokens.append(merge)
								idx += 2
								continue
							else:
								new_tokens.append(old_tokens[idx])
								idx += 1
					old_tokens = new_tokens

				self.pre_token_cache[tokens] = tuple(old_tokens)
				output_tokens.extend(old_tokens)
		return output_tokens
	
	def encode_iterable(self, iterable):
		for block in iterable:
			tokens = self.encode(block)
			yield from tokens
	
	def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens=None):
		pass

	def decode(self, ids: list[int]) -> str:
		byte_string = b''.join([self.vocab[id] for id in ids])
		return byte_string.decode(encoding="utf-8", errors="replace")
	
if __name__ == "__main__":
	# vocab, merges = train_bpe_tokenizer("./data/TinyStoriesV2-GPT4-valid.txt", vocab_size=20000, special_tokens=["<|endoftext|>"], num_processes=10)

	# tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
	
	# with open("./data/TinyStoriesV2-GPT4-train.txt") as f:
	# 	text = f.read()
	# text = text[:int(1e7)]
	# encode = tok.encode(text)
	# decode = tok.decode(encode)
	# print(text == decode)

	import json
	import os
	import resource
	import sys

	import psutil
	import pytest
	import tiktoken
	import time

	from tests.adapters import get_tokenizer
	from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

	from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

	VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
	MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

	VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
	MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

	# TEST 1
	reference_tokenizer = tiktoken.get_encoding("gpt2")
	tokenizer = get_tokenizer_from_vocab_merges_path(
		vocab_path=VOCAB_PATH,
		merges_path=MERGES_PATH,
		special_tokens=["<|endoftext|>"]
	)

	with open("./data/TinyStoriesV2-GPT4-train.txt") as f:
		text = f.read()
	# text = text[:int(1e9)]
	
	print("starting tiktoken")
	t0 = time.time()
	encode2 = reference_tokenizer.encode(text, allowed_special={'<|endoftext|>'})
	t1 = time.time()
	total = t1-t0
	print(f"tiktoken time: {total}")

	print("my tokenizer")
	t0 = time.time()
	encode = tokenizer.encode(text)
	t1 = time.time()
	total = t1-t0
	print(f"My tokenizer time: {total}")


	






	# corpus_path = FIXTURES_PATH / "address.txt"
	# with open(corpus_path) as f:
	# 	corpus_contents = f.read()
	
	# # corpus_contents = corpus_contents[:1500]
	# reference_ids = reference_tokenizer.encode(corpus_contents)
	# ids = tokenizer.encode(corpus_contents)
	# assert ids == reference_ids

	# assert tokenizer.decode(ids) == corpus_contents
	# assert reference_tokenizer.decode(reference_ids) == corpus_contents

	# TEST 2
	# tokenizer = get_tokenizer_from_vocab_merges_path(
	# 	vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
	# )
	# test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
	# encoded_ids = tokenizer.encode(test_string)
	# tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
	# decoded_string = tokenizer.decode(encoded_ids)
	# # Ensure the special <|endoftext|> token is preserved
	# print(tokenized_string)
	# print(decoded_string)
	# print(test_string)
	# assert tokenized_string.count("<|endoftext|>") == 3
	# assert test_string == decoded_string
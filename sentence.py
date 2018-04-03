class _Sentence:
	def __init__(self, content):
		self.word_list = []
		self.num_words = 0
		self.raw_content = content

	def __str__(self):
		return str(self.wordList) + ('*' if self.num_words == 0 else '')

	def addWord(self, word_index):
		self.word_list.append(word_index)
		self.num_words += 1
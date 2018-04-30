class _Document:
	def __init__(self):
		self.num_sentences = 0
		self.sentence_list = []

	def __str__(self):
		return '%d: {%s}'%(self.num_sentences, ', '.join(str(para) for para in self.sentence_list))

	def add_sentence(self, sentence_obj):
		self.sentence_list.append(sentence_obj)
		self.num_sentences += 1

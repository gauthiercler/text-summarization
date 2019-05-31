'''
from rouge import FilesRouge
#Is this the right way to make class..?
class RougeEval:
	def __init__(self):
		self.sumfile="summary.txt"
		#our summary
		self.reffile="reference.txt"
		#ideal summary
	def run(self):
		self.files_rouge = FilesRouge(sumfile , reffile)
		self.scores = files_rouge.get_scores()
		
		print(self.scores)

'''
from rouge import FilesRouge
#Is this the right way to make class..?
class RougeEval:
	sumfile="summary.txt"
		#our summary
	reffile="reference.txt"
		#ideal summary
	files_rouge = FilesRouge(sumfile , reffile)
	scores = files_rouge.get_scores()
		
	print(scores)
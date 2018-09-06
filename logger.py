import logging
import datetime
import random


class Singleton(type):
	_instances = {}

	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]


class TimeHandler(metaclass=Singleton):
	def __init__(self):
		self.now = datetime.datetime.now()
		self.h_file_name = \
			'TableTrainNet-' + \
			str(self.now.year) + "-" + \
			str(self.now.month) + "-" + \
			str(self.now.day) + "--" + \
			str(self.now.hour) + "-" + \
			str(self.now.minute) + "-" + \
			str(self.now.second) + "--" + \
			str(random.randint(0, 10000)) + '.log'
		self.handler = self.return_handler(self.h_file_name)

	@staticmethod
	def return_handler(log_name):
		# create a file handler
		handler = logging.FileHandler(log_name)
		handler.setLevel(logging.INFO)

		# create a logging format
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		return handler

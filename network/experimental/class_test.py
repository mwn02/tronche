class Person():
	def __init__(self) -> None:
		self.health = 50

	def heal(self, amount):
		self.health += amount

person1 = Person()
person1.heal(30)

person2 = Person()
person2.heal(50)
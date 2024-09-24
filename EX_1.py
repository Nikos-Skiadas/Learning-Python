

class Person:
    
    count = 0	# Variable to track the number of Person objects created

    def __init__(self, name, age, reg_number):
        self.__name = name
        self.__age = age
        self.__reg_number = reg_number
        Person.count += 1  # Increment the counter when a new object is created

    def __del__(self):
        Person.count -= 1  # Decrement the counter when an object is deleted
        print(f"Person object with registration number {self.__reg_number} has been deleted.")

    def get_name(self):
        return self.__name

    def set_name(self, name):
        self.__name = name

    def get_age(self):
        return self.__age

    def set_age(self, age):
        self.__age = age

    def get_reg_number(self):
        return self.__reg_number

    def set_reg_number(self, reg_number):
        self.__reg_number = reg_number
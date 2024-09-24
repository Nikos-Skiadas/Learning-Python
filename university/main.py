class Name:

    def __init__(self, first: str, last: str):
        self.first = first
        self.last = last

    def __repr__(self) -> str:
        return f"{self.first} {self.last}"


class Email:

    def __init__(self, user: str, host: str):
        self.user = user
        self.host = host

    def __repr__(self) -> str:
        return f"{self.user}@{self.host}.com"


class Person:

    count = 0	# variable to track the number of Person objects created


    def __init__(self, name: str, reg_number: str, email: str):
        self._name: Name = Name(*name.split())
        self._reg_number = reg_number
        self._email: Email = Email(*email.strip(".com").split("@"))

        Person.count += 1  # increment the counter when a new object is created

    def __del__(self):
        Person.count -= 1  # decrement the counter when an object is deleted


    @property
    def name(self) -> Name:
        return self._name

    @property
    def reg_number(self) -> str:
        return self._reg_number

    @reg_number.setter
    def reg_number(self, reg_number: str):
        self._reg_number = reg_number

    @property
    def email(self) -> Email:
        return self._email


if __name__ == "__main__":
    x = Person(
        "Nikos Skiadas",
        "AM10203401",
        "nikos.skiadas@uoa.com",
    )
    x.name.first = "Stratos"
    x.name.last = "Papadoudis"

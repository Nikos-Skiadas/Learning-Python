from __future__ import annotations


class Name:

    def __init__(self, first: str, last: str):
        self.first = first
        self.last = last

    def __repr__(self) -> str:
        return f"{self.first} {self.last}"


class Email:

    @classmethod
    def from_name(cls, name: Name, host: str) -> Email:
        return Email(f"{name.first}.{name.last}", host)


    def __init__(self, user: str, host: str):
        self.user = user
        self.host = host

    def __repr__(self) -> str:
        return f"{self.user}@{self.host}.com"


class Person:

    count = 0	# variable to track the number of Person objects created


    def __init__(self, name: str, reg_number: str, host: str):
        self._name = Name(*name.split())
        self._reg_number = reg_number
        self._email = Email.from_name(self._name, host)

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
        "physics.uoa",
    )
    x.name.first = "Stratos"
    x.name.last = "Papadoudis"

"""Exercises:

Add a `department` attribute to `Person`, that is a string with the department name.
Maybe each department is formatte as 'Department of <name>'.

Update the `Person` constructor to use the department as well, to generate a full email adress.
This means that the user depends on the person name, and the host depends on the department.

Bonus: Start working on a `Department` class.
If this class is made, what should happen to the string `department` attribute of person?
Should it now be a `Department` object?
If yes, what changes to the email creation for a person?
"""


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


    def __init__(self, name: str, reg_number: str, host: str, department: str):
        self._name = Name(*name.split())
        self._reg_number = reg_number
        self._email = Email.from_name(self._name, host)
        self._department = f"Department of {department}"
        
        email_host = department.lower()
        self._email = Email.from_name(self._name, email_host)

        Person.count += 1  # increment the counter when a new object is created

    def __del__(self):
        Person.count -= 1  # decrement the counter when an object is deleted


    @property
    def name(self) -> Name:
        return self._name

    @property
    def reg_number(self) -> str:
        return self._reg_number

    @property
    def email(self) -> Email:
        return self._email
    
    @property
    def department(self) -> str:
        return self._department


if __name__ == "__main__":
    x = Person(
        "Nikos Skiadas",
        "AM10203401",
        "physics.uoa",
        "NTUA"
    )
    x.name.first = "Stratos"
    x.name.last = "Papadoudis"

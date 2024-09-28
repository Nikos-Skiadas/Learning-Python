"""Exercises:

Add a `department` attribute to `Person`, that is a string with the department name.
Maybe each department is formatted as 'Department of <name>'.

Update the `Person` constructor to use the department as well, to generate a full email adress.
This means that the user depends on the person name, and the host depends on the department.

Bonus: Start working on a `Department` class.
If this class is made, what should happen to the string `department` attribute of person?
Should it now be a `Department` object?
If yes, what changes to the email creation for a person?
"""


from __future__ import annotations


import string


class Name:

    def __init__(self, first: str, last: str):
        self.first = string.capwords(first)
        self.last = string.capwords(last)

    def __repr__(self) -> str:
        return f"{self.first} {self.last}"


class Email:

    def __init__(self, user: str, department: Department):
        self.user = user.lower()
        self.host = department.domain

    def __repr__(self) -> str:
        return f"{self.user}@{self.host}.com"


class Person:

    count = 0  # variable to track the number of Person objects created


    def __init__(self, name: str):
        self.name = Name(*name.split())
        self._department: Department | None = None

        Person.count += 1  # increment the counter when a new object is created

        self.count = Person.count

    @property
    def department(self) -> Department | None:
        return self._department

    @department.setter
    def department(self, department: Department):
        self._department = department
        self.registry = f"{department.domain.split('.')[0]}{self.count:06d}".upper()


    @property
    def email(self) -> str:
        return str(Email(self.registry, self.department)) if self.department is not None else ""


    def __repr__(self) -> str:
        return f"{str(self.name):32} {str(self.email):32} {self.department.name if self.department is not None else '':32}"

    def __del__(self):
        Person.count -= 1  # decrement the counter when an object is deleted


class Department:

    def __init__(self, name: str, university: str):
        self.name = f"Department of {string.capwords(name)}"
        self.domain = f"{''.join(word[0] for word in name.split())}.{''.join(word[0] for word in university.split())}".lower()

        self.persons = dict[str, Person]()

    def __repr__(self) -> str:
        return "\n".join(str(person) for person in self.persons.values())

    def __iadd__(self, person: Person) -> Department:
        person.department = self
        self.persons[person.registry] = person

        return self


if __name__ == "__main__":
    informatics = Department(
        "Computer Science",
        "University of Athens",
    )
    yoda = Person("Nikos Skiadas")
    vader= Person("Stratos Papadoudis")

    informatics += yoda
    informatics += vader

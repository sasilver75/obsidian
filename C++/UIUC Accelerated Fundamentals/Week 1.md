

We're going to be using C++ 14 in this course

Here's some language references
- [http://www.cplusplus.com/](http://www.cplusplus.com/)
- [https://en.cppreference.com/w/](https://en.cppreference.com/w/)

Here's a different tutorial to c++ that you might like to also have:
- [https://cplusplus.com/doc/tutorial/](https://cplusplus.com/doc/tutorial/)
  
Here's a definitive list of C++ books that SO things are good:
- [https://stackoverflow.com/questions/388242/the-definitive-c-book-guide-and-list](https://stackoverflow.com/questions/388242/the-definitive-c-book-guide-and-list)

Running C++ on a Mac:
https://courses.engr.illinois.edu/cs225/sp2019/guides/own-machine/#macos

Here's a cheat sheet to LLDB, which is the debugger that comes with XCode on macs:
https://gist.github.com/ryanchang/a2f738f0c3cc6fbd71fa

`make` will look for a Makefile in the same directory, and it will follow the carefully scripted instructions there to do what we need. You will see it list diagnostic messages on the terminal when it does that.


C++ Classes encapsulate data and associated functionality into an object
```c++
class Cube {
	public:
		double getVolume();
		// ...
	private:
		double length_;
};
```

==Encapsulation== encloses data and functionality into a single unit, called a ==class==.
- Think of it like a pill! Inside the pill, you have both data and functionality.
	- Data: The length of the side of a cube
	- Functionality: Getting the volume of the cube from our private variables.

When we have private member variables of a class, we're going to follow the pattern of adding a trailing underscore, so that it's more obvious.

In c++, functionality and data are separated into two separate ==protections== : 
- private data + functionality
- public data + functionality

These guide who can access what ==members== of our class.
- Public members CAN be accessed by client code
- Private members CANNOT be accessed by client code (only used within the class itself)

In C++, the ==interface== (.h file) to the class is defined separately from the ==implementation== (.cpp file).

Let's look at a header file.

A header file (.h) defines the interface to the class, which includes:
- Declaration of ALL member variables
- Declaration of ALL member functions
(It includes the signatures, but not the implementations of these members. So it's sort of like the API documentation to a class)


Cube.h
```cpp
#pragma once

class Cube {
	public:
		double getVolume();
		double getSurfaceArea();
		void setLength(double length);
	
	private:
		double length_;
};

```
- The `#pragma once` line of code is always present in our header files.
	- It instructs compilers to only compile our file once.
- Notice that the class always ends with a semicolon
- Notice the curly braces
- Notice the public and private ==protection regions==
	- Containing public variables and public functions and the private variables and private functions
- Note that this is just the API to our classs, it doesn't have any implementation.

Let's look at our implementation file:

Cube.cpp
```cpp
#include "Cube.h"

double Cube::getVolume() {
	return length_ * length_ * length_;
}

double Cube::getSurfaceArea() {
	return 6 * length_ * length_;
}

void Cube::setLength(double length) {
	length_ = length;
}
```
- Notice that we "include" our header file for this class
- Notice that we're implementing the functions defined in our header file.
- The functions name need to include the class in which they were defined; This is the Cube:: bit.
- We then define the implementation of the Cube's getVolume function. You can see that this function is able to reference the (in this case private) member variables on the class.

Now we can go ahead and build a program using our cube class.

main.cpp
```cpp
#include <iostream>
#include "Cube.h"

int main() {
	Cube c;
	c.setLength(3.48);
	double volume = c.getVolume();

	std::cout << "Volume: " << volume << std::endl;

}

```
- Notice that our main file that makes use of Cube is only actually including the header file. And recall that the header file doesn't explicitly  even include or point to the implementation cpp file.
	- including the header file lets the Cpp compiler know what a cube is.
- Recall that the main() function needs to be called main() and that there DOESN'T need to be a semicolon after the function declaration curly braces close (compare this with the class declaration).
- See in the function that we're creating a new cube, setting its length, getting the volume and storing it in a volume variable (double-typed). then we finally cout (or console out) the volume of our cube, using the concatenation operator (and the endline).


The C++ standard library (std) (also sometimes called the cpp standard template library, or stl)
- It's divided into a number of separate sub-libraries that can be `included`'d 
- The `iostream` library allows us to output data to both files and to the console itself, like using std::cout
	- Must include `#include <iostream>`

```cpp
#include <iostream>

int main() {
 std::cout << "Hello, world!" << std::endl;
 return 0;  // Success code
}

```

Nice! So by using cout and including iostream, we can output anything to the console!

Standard library organization
- Anything in the standard library will be part of the `std` namespce
	- ==Namespaces== allow us to avoid name conflicts for commonly used names

```cpp
#include <iostream>

using std::cout;
using std::endl;

int main() {
	cout << "hello world" << endl;
	return 0;
}

```
- Here, we're using the `using` keyword so that we can just refer to cout and endl as their names (without the namespaces) lower in the code. This **imports that feature into the global space**.
	- This makes your code harder to read, in some senses, so it's usually only used for a few very specific functions (eg cout). It's better not use this feature otherwise.

Let's revisit our cube!****

We can choose whatever namespace name we want for our cube; We just can't start with a number, and some other deatils:
```cpp
#include <iostream>
#include "Cube.h"

int main() {
	uiuc::Cube c;
	c.setLength(2.4);
	std::cout << "Volume: " << c.getVolume() <<< std::endl;

	double surfaceArea = c.getSurfaceArea();
	std::cout << "Surfaec Area: " << surfaceArea << std.endl;

	return 0;
}

```
- Note that it seems like you have to have explicit returns in our case; if you don't have an explicit return, your function will return an undefined value.

We placed the cube into the uiuc namespace; how did we do that?

A "cube" is rather generic; you could imagine that there could be many cubes out there across libraries; we want to specify that our cube is different from other cubes that are out there! We want to put it in some namespace `uiuc` that's separate from any other namespace out there!

Cube.h
```cpp
#pragma once

namespace uiuc{
	class Cube{
		public:
			...
		private:
			...
	}; // Recall that you need a semicolon afterwards
}

```

Cube.cpp
```cpp
#include "Cube.h"

namespace uiuc {
	double Cube::getVolume(){...}
	double Cube::getSurfaceArea(){...}
	double Cube::setLength(double length){...}
}

```

That's it! We just wrap the class in the header file and the function implementations in the implementation file with a `namespace <namespacename>`! 


---

Week 1 Quiz (Just the answers):
- Every variable in C++ has to be associated with a specific type
	- C++ is "strongly typed", meaning that the type of every variable is assigned when the variable is declared, and the type of a variable CANNOT CHANGE after the variable is declared!
- According to the C++ standard, main() is the name of the function that's the starting point for a program
	- The program begins when the operating system calls the function "main()"
- A class can consist of multiple member data variables of different types, but each type must be specified when the class is defined.
- The member functions of a class always have access to every member data variable (public or private protection group) of that class.
- The `#include` DIRECTIVE is used to insert the contents of another file at the current location while processing the current file.
- To instantiate an instance of a Pair class that's within the uiuc namespace, we would do: `uiuc::Pair p;`
- The `using` keyword is used to indicate the namespace to search through to find classes and variables when they are later referenced (sans namespace) throughout the rest of the program.
- We use namespaces in C++ programming because two libraries might use the same label for a class or variable.
- What is the namespace of the C++ Standard Library?
- The `<<` (`Streaming`) operator is used to send a sequence of strings, numbers, and other values to the standard library's cout object. 
	- Note that std::cout << "a" << 3;
		- is first evaluated as (std::cout << "a") << 3;
		- The expression in the parentheses returns a reference to COUT after sending "a" to it, so that the second "<<" operator sends the value to cout. This will be useful later!
Let's now focus on the final bits of C++ that we'll use in rest of the course.

Let's discover the Tower of Hanoi program, which is a program to help us solve a pretty complicated problem!


#### Template Types
-  A template type is a special type that can take on different types when teh type is initialized! std::vector uses a template type!
```cpp
std::vector<char>
std::vector<int>
std::vector<uiuc::Cube>
```
All of these things represent an array of objects in memory, and it's done through a standard, single library called the vector!
- The std::vector class is part of the Standard Template Library (STL) we mentioned before; it's the functionality of a dynamically growing array with a "templated" type

Key ideas:
- Defined in: `#include <Vector>`
- Initialization: `std::vector<T> v;`
	- We give the type that we want to have in the vector in the T
- Add to (back) of array: `::push_back(T)` , eg `v.push_back(4)`
- Access specific element: `::operator[](unsigned pos);` , eg `v[2]`
- Number of elements: `::size()`

Let's see an example:

```cpp
#include <vector>
#include <iostream>

int main() {
	// We don't know how long this vector is going to be, when we initalize it. We just know that it's going to have some ints in it.
	std::vector<int> v:
	v.push_back(2); // We push 2 to the end of the vector ("append")
	v.push_back(3);
	v.push_back(5);

	std::cout << v[0] << std::endl;
	std::cout << v[1] << std::endl;
	std::cout << v[2] << std::endl;

	return 0;
}
```

Here's another

```cpp
#include <vector>
#include <iostream>

int main() {
	std::vector<int> v;

	// 100 times, pushing the square of each i
	for (int i = 0; i < 100; i++) {
		v.push_back(i*i)
	}

	std::cout << v[12] << std::endl; //144
	
	return 0;
}

```
So we now have an ability to create an array-like structure in C++ called a Vector. A vector is slightly different from an array that you might be familiar with, because it's dynamically sized, and grows/shrinks as needed.

Let's build something with it.

#### Tower of Hanoi - Intro

![[Pasted image 20231218203154.png]]
We can visualize each of positions as a stack, and each stack containing cubes.
This entire thing (the stacks) might be encapsulated in something called a game
![[Pasted image 20231218203539.png]]

Let's see a cube class:
```cpp
namespace uiuc {
	class Cube {
		public:
			// Our cube constructor takes a length and a color
			Cube(double length, HSLAPixel color);
			
			double getLength() const; // What does this const do here? The const keyword after a function declaration has a specific meaning, indicating that the member function is a COSNTANT member function, meaning that it DOES NOT MODIFY any member variables of the class (except those marked as mutable). This providse a guarantee that calling the function will not alter the state of the object. In this way, const is commonly used for accessor/"getter" functions, which return values without modifying the object.
			void setLength(double length);

			double getVolume() const;
			double getSurfaceArea() const;

		private:
			// Internally, we have private member variables for our length and color. The _ trailing is a stylistic thing that denotes "private", but it's not required afaict.
			double length_;
			HSLAPixel color_;
	}
}

```

Okay, those are our cubes!
We don't know how many cubes there are going to be. But now we need to make some sort of Stack class to house these cubes!
- Because we don't know, a dynamic array or a Vector in C++ is a great way to represent this!
- Each stack will have a vector of cubes.

In addition to the vector of cubes (state), we also need some operations to interact with the top of the stack (functionality). Recall that classes are all about containing state and controlling functionality/access to that state.

```cpp

class Stack {
	public:
		// Notice that we only have an automatic default constructor?
		void push_back(const Cube& cube); // Takes a const reference to a cube; we won't modify the cube, because it's a const.
		Cube removeTop(); // Returns a copy of the cube on the top
		Cube& peekTop();  // Returns a reference to a cube (but doesn't remove it from our internal data structure)
		unsigned size() const; // "unsigned" here is shorthand for unsigned int. The const after the function name indicates that we won't be changing the member variables in our instance.

		// An overloaded operator<< which lets us print the stack via cout<<
		// He calls ostream& "ostream by reference"
		// We'll see this quite a bit throughout this course! 
		// The friend ostream thing lets us cout an object. It's something like the tostream function; it lets us output an object in a textual way.
		friend std::ostream& operator<<(std::ostream& os, const Stack& stack);

	private:
		std::vector<uiuc::cube> cubes_; // Some cubes!

}

```

Now let's build the functionality for the game itself
- Sets up the array of 3 stacks
- Initial state has four cubes in the first stack

Game.h header file
```cpp
#pragma once
class Game {
	public:
		Game(); // Will have a custom default constructor that will initialize our stuff
		void solve(); // This will... solve the game, updating the state in this Game?

		friend std::ostream& operator<<(std::ostream& os, const Game& game);

	private:
		std::vector<Stack> stacks_; // A vector of stacks, each of which will have some Cubes!
}

```

Game.cpp implementation file
```cpp
...


Game::Game() {
	// Create three empty stacks
	for (int i = 0; i < 3; i++) {
		Stack stackOfCubes; // Use the automatic default constructor for the stack
		stacks_.push_back(stackOfCubes) // Push back (append) them to the Game's vector of stacks
	}

	// Now create four differently-sized-and-colored blocks and put them in the first stack in the game's vector of stacks
	Cube blue(4, uiuc::HSLAPixel::BLUE);
	stacks_[0].push_back(blue);

	Cube orange(3, uiuc::HSLAPixel::BLUE);
	stacks_[0].push_back(orange);

	Cube purple(2, uiuc::HSLAPixel::BLUE);
	stacks_[0].push_back(purple);

	Cube red(1, uiuc::HSLAPixel::BLUE);
	stacks_[0].push_back(red);

	// So the game has the stacks and the first stack has the cubes!
}

// Now we would want to implement the solve method!

```

In main.cpp
```cpp
#include "Game.h"
#include <iostream>

int main() {
	Game g; // This calls the primary default constructor of game, which creates the stacks, and puts four cube in the first stack

	std::cout << "Initial game state: " << g << std::endl;

	g.solve();

	std::cout << "Final game state: " << g << std::endl;

}

```


....Tower of Hanoi Solutions....

### Templates and Classes
- C++ lets us use the power of templates in building our classes (eg `vector<int>`).
	- A ==templated variable== is declared by using a special syntax before either the beginning of a class or a functino!

```cpp

// Applying to a class
template <typename T>
class List {
	...
	private:
		T data_;
}

// Appliny got a function
template <typename T>
int my_max(T a, T b) { 
	if (a > b) { 
		return a; 
	}
	return b;
}

```

These templated variables are checked at COMPILE TIME, which allows errors to be caught before running to program
```cpp
my_max(4, 7); // This workS!
my_max(Cube(3), Cube(6)); // results in a compiler error, since our cubes don't have the ability to be compared >. We haven't overloaded the > operator for our cube class yet!
```

Let's see:
```cpp
template <typename T>
T max(T a, T b) { 
	return (a > b) { return a; }
	return b
}

int main() {
	max(3,5); // 5
	max(std::string("Hello"), std::string("World")); // World. We use this std::string instead of just teh string literals, because the standard library lets us compare these std::string objects by alphabetical order using ">". However if we directly compare the string literals of "Hello" and "World", we'll end up simply comparing the addresses of two arrays of const chars. That's not what we want
	max(Cube(3), Cube(6)); // This is going to have a compilation error because we haven't defined a way to compare these cubes.
	return 0;
}
```

#### Inheritance

- It's a powerful concept in C++ that lets us inherit all of the member functions and data from a ==base class== into a ==derived class==
- A base class is a generic form of a specialized, dervied class
	- Eg a Cube is a specialized version of a Shape. Perhaps every 2d (or 1d) shape has a *length*, and that *length* member variable could be shared amongst 

Shape.h
```cpp
#pragma once

// Notice that we don't have getSurfaceArea or getVolume; these are specific to our cube!
class Shape {
	public:
		Shape(); // Custom Default Constructor
		Shape(double width); // Custom default constructor
		double getWidth() const; // A "getter" (marked with const because it doesnt' modify anything) fn
	private:
		double width_;
}

```

In our Cube class header file:
```cpp
#pragma once

#include "Shape.h"
#include "HSLAPixel.h"

namespace uiuc {
	class Cube : public Shape { // This is how we declare that Shape is our base class!
	// We'd read this as: "The class Cube inherits from the class Shape." Everything we do in this class will be public inheritance; 99% of the use of inheritance is public inheritance.
		public:
			Cube(double width, uiuc::HSLAPixel color); // Our constructor. Notice that it takes the things required to make a Shape as well
			double getVolume() const;  // Cube-specific functionality

		private:
			uiuc::HSLAPixel color_; // Only our cube has a color, not a shape. The width of orur cube is 
	}
}

```

#### Initialization of a class base class
- When a derived class is initialized, the derived class MUST construct the base class!
	- `Cube` must construct `Shape`
	- By default, the Cube uses the Shape's default constructor
	- Custom constructors can be used with an ==initialization list==

Cube.cpp
```cpp
#include "Cube.h"
#include "Shape.h"

namespace uiuc {
	// A custom constructor of a derived class!
	Cube::Cube(double width, uiuc::HSLAPixel color) : Shape(width) { // Because we want a shape with a specific width, we use the initialization list syntax! This is the colon followed by the name of the base class. We're asking C++ to initialize the Shape class using teh custom constructor that takes the width as a parmetre.
		// the first thing that happens is that shape gets constructed using teh width
		color_ = color;
	}

	double Cube::getVolume() const {
		// Our getter function to get volume. Recall that our Volume is determined by our width.
		// But we can't access Shape::width_ directly, due to it being in the `private` protection region!
		// Instead, we have to use the public Shape::getWidth(), a public function. If you think about it from an OO perspective, this is a good thing that stops you from doing bad things.
		return getWidth() * getWidth() * getWidth();
	}

}

```

Access Control
- When a base class is inherited, the derived class:
	- CAN access all public members of the base class
	- CANNOT access private members of the base class

That initializer list syntax that we used before is used quite a lot in C++

==Initializer Lists==
- The syntax to initialize the base class is called the ==initializer list==, and can be used for several purposes:
	- Initialize a base class (as we did above)
	- Initialize the *current class* using *another constructor*
		- If we want to delegate all the work to a different constructor
	- Initialize the default values of member variables

So Initializer lists allow us to do more than just initialize base classes

Cube.cpp
```cpp
#include "Shape.h"

// We're defining a custom default constructor for our Shape class that actually uses a differ custom non-default constructor that we've already defined!
Shape::Shape() : Shape(1) {
	// Nothing in here
}

// We have an initializer list here that initializes the private member varibale width to the value passed in as the width. 
Shape::Shape(double width) : width_(width) {
	// Nothing
}

double Shape::getWidth() const {
	return width_;
}
```


-----

Week 4 Quiz

![[Pasted image 20231218231319.png]]
- Template types allow you to create functions and classes that don't have all of their types explicitly defined, but I don't think that they apply to local variables. I'm guessing this is the problematic one.
- This one seems to be to be quite similar to the first one... But it seems like it's probably more reasonable.
- Okay, this one is template functions. We can do that.
- This is basically the same template functions thing, but it's for "methods" (which are just member functions in C++). We can do this.



![[Pasted image 20231218231407.png]]
- B


![[Pasted image 20231218231413.png]]
- D


![[Pasted image 20231218231419.png]]
- You just call it normally. The template doesn't affect how you invoke the function.

![[Pasted image 20231218231427.png]]
- A is just passing in two doubles -- we can do this.
- B is passing in two different types... but it's possible that the string literal "five" is actually interpreted as some large number (eg the address of the array of chars)...
- C is trying to compare a Just_a_double class an an int. This is going to break, surely.
- D we haven't defined comparator operator overloads for our Just_a_double class.

![[Pasted image 20231218231423.png]]
- A


![[Pasted image 20231218231640.png]]
- **D** is the only one that uses the correct syntax to be a derived class, I think... But it doesn't look like it uses the "public" keyword before Pair, which we're used to seeing. Maybe it's not needed, though? ⬅️
- It's definitely not B
- It's ... probably not C either, since there's not reference to really being part of/contained by our equalPair class...
- I don't think it's A, I'm not familiar with that syntax. What is `this` again? `this` is a pointer to the instance of the class in which it appears. However in this case, with the attempt to call the constructor o the Pair class... you cannot call a constructor directly like this after the object has been created; Constructors are only called when an object is first created, not afterward.



![[Pasted image 20231218231650.png]]
- The derived class's function will not have access to the private member variables or functions of the base class.
- This means that "Just the member variable isEqual of equalPair" is the answer
	- The derived class equalPair still has access to its own private members!


![[Pasted image 20231218231654.png]]
-  It's fine that Just_a_double the default custom constructor is deferring using an initializer list.
- Both of these actually work! This is called a member initialization list in C++. This is often the preferred way to initialize members, especially for types that don't have a default constructor, or when you want to initialize a member with a value provided as an argument to the constructor.

![[Pasted image 20231218231700.png]]
- B 






Let's learn about object oriented programming in C++ by learning about the lifecycle of objects in C++!

When an instance of a class is created in C++, the class constructor lets us program EXACTLY the initial state of the object that's created.

==Automatic Default Constructors== are provided for your classe for free!
The automatic default constructor will only initialize all member variables to their default values:


```cpp

#pragma once

namespace uiuc{
	class Cube{
		public:
			double getVolume();
			double getSurfaceArea();
			void setLength(double length);
		private:
			double length_;
	}

}


```
- There's no constructor in this class, so we get an automatic default constructor, which initializes all variables to their DEFAULT VALUES; so length_ will take on its default value.
	- The default value for a primitive type is largely undefined; we aren't sure what he length will be!
	- If our private member variable length_ were a class, then the constructor of that class would define what a default value is.

==Custom Default Constructors==
- The simplest constructor we can provide is a custom default constructor that specifies the state of the object when the object is constructed.
- We define one by creating:
	- A member function with the same name of the class itself
	- The function takes zero parameters
	- The function does not have a return type

```cpp
Cube::Cube() //This overrides the default constructor that C++ provides for us; it 
```
The constructor builds the object but doesn't return anything back to the program.

```cpp
#pragma once

namespace uiuc {
	class Cube {
		public:
			Cube(); // custom default constructor!
	
			double getVolume();
			double getSurfaceArea();
			void setLength(double length);
		private:
			double length_;
	}
}

```
We define the implementation of the Cube custom constructor in our cpp implementation file:

```cpp
namespace uiuc {
	Cube::Cube() {
		length_ = 1;
	}
}
```

We then in our main app
```cpp
#include "Cube.h"  // Including a file in the same directory by name
#include <iostream>  // Including a file in that ... lookuppy path.

int main() {
	uiuc::Cube c;  // Note that we just ... declare it. We don't have to invoke it. We don't have to do "new c"
	std::cout << "Vol: " << c.getVolume() << std::endl;  // Should say "Vol: 1"
	return 0;
}

```

We just talked about a Custom Default Construct ^ which zero parameters

But we can also define non-default Custom Constructors (or just Custom Constructors) that let user code specify arguments to our constructor!
- We can let a user specify the length of the cube when they create the cube!

We can have multiple constructors:
Header file:
```cpp
#pragma once

namespace uiuc {
	class Cube {
		public:
			Cube(); // Our CUSTOM default constructor with zero parameters (cf with the automatic d.c.)
			Cube(double length); // A custom NON-DEFAULT constructor that takes arguments
		
			double getVolume();
			...
		private:
			...
	}
}
```

And then in our implementation file:
```cpp
namespace uiuc {
	Cube::Cube() {
		length_ = 1
	}

	Cube::Cube(double length) {
		length_ = length;
	}
}
```

In our main.cpp
```cpp
...

int main() {
	uiuc::Cube c(2);  // We invoke it (cf with last time with the custom default constructor)
	std::cout << "Vol: " << c.getVolume() < std::endl;  // Vol: 8
	return 0;
}

```

==NOTE==:
If ANY custom constructor at all is is defined, an automatic default constructor is NOT defined.

Let's look at an eaxmple of this coming into play:

Header
```cpp
#pragma once

namespace uiuc {
	class Cube {
		public:
			Cube(double length); // Recall that this is how we define a custom non-default constructor
			// Above: Ours is goin to take arguments. So because we did this, we DO NOT have an automatic default constructor created for us!

			double getVolume();
			double getSurfaceArea():
			void setLength(double length);
		private:
			double length_;
	}
}
```

In our main.cpp:
```cpp
#include "Cube.h"
#include <iostream>

int main() {
	uiuc::Cube c; // !! This is going to fail! This tries to use a default constructor, but there isn't one!
	...
}
```
We get a compiler error saying: The candidate constructor is not viable! Requires single argument 'length' but not arguments were provided.

---
Aside:
- the professor is using the Clang compiler (clang++ to be specific). Some sessions of this course tell you to install GCC instead (this is the GNU Compiler Collection, launched by g++ in the command line). These compilers are similar but the error messages and warnings they display aren't going to be the same.
----

Let's talk about another form of constructor that helps us copy one instance of an object to another instance of an object.

### Copy Constructors

In addition to standard constructors, copy constructors are a special type of constructor that let us make a copy of an existing object.
- If we don't provide a custom copy constructor, the C++ compiler provides an ==automatic copy constructor== for our class for free!
	- The automatc copy constructor will copy the contents of all member variables
		- This might be what you want, but you can define a custom one!

Custom Copy Constructoor
- It must
	- Be a class constructor 
		- Follow all the rules of a class constructor
	- Have exactly one argument
		- This must be a const reference of the same type as the class itself
		- A constant reference is like this:L
```cpp
Cube::Cube(const Cube & obj)
```

Let's look at some code:

```cpp
#include "Cube.h"
#include <iostream>

namespace uiuc {
	// This is a custom default constructor (an automatic default constructor is not made)
	Cube::Cube() {
		length_ = 1;
	}

	// This is a custom copy constructor! (an automatic copy constructor is not made)
	Cube::Cube(const Cube & obj) {
		// This takes in an object, and the goal is to copy the contents of the object into our current instance. 
		// Since we only have a single member variable, the below is all we have to do.
		// The automatic copy constructor would be doing exactly this, so we don't need this, but as we go on, we'll need to do custom copy constructors!
		length_ = obj.length_;
	}
	
}

```

- Copy Constructor Invocation
	- Often, copy constructors are invoked automatically!
		- Passing an object as a parameter (by value)
		- Returning an object from a function (by value)
		- Initializing a new object from a previous version of an object.

So that we can actually understand what's going on...

```cpp
#include "Cub.h"
#include <iostream>

namespace uiuc {
	// Her'es a default custom constructor
	Cube::Cube() {
		length_ = 1;
		std::cout << "Default constructor invoked!" << std::endl;
	}

	// Here's a copy constructor
	Cube::Cube(const Cube & obj) {
		length_ = obj.length_;
		std::cout << "Copy constructor invoked!" << std::endl;
	}
}

```

So when we run our program, we'll see exactly what code is running when we have a set of source code!

main.cpp
```cpp
...

void foo(Cube cube) {
	// Nothing
}

int main() {
	Cube c; // invokes defautl constructor
	foo(c); // Foo takes in a cube as an argument. Becaues this arg is being apssed into a function, it has to be copied from main into the foo stack frame. So we expect a call to the copy constructor to do this!
}
```

When we run it..
```
Default constructor invoked!
Copy constructor invoked!
```

Hell yeah.


What if we look at another example:

```cpp
#include "../cube.h"
using uiuc::Cube;

Cube foo() {
	Cube c; //Default csontructor called
	return c; // Copy constructor called [after]
}

int main() {
	Cube c2 = foo(); // We have an object in foo, need to copy it over to main so that main can make use of it. Now that i's back in miain, we need to do something. We see that c2 needs totake on the value of what's returneed by foo. so we have a third copy consstructor that copies from the main stack frame into the variable c2.
	return 0;
}

```
DefaultConstructor: Create the cube
CopyConstructor: Return to main stack frame
CopyConstructor: Put it into c2

Another example:
```cpp
...

int main() {
	Cube c; // Calls Default Cosntructor
	Cube myCube = c;  // Because we're initializing this variable, and it's taking on the value of c, the avlue of c is copied into myCube. so expect the Copy Constructor to run
}

```
DefaultConstructor
CopyConstructor

Another Example!
```cpp
include "../Cube.h"
using uiuc::Cube;

int main() {
	Cube c; // default constructor
	Cube myCube; // default constructor

	myCube = c; // Both myCube and c have both alreay been constructed by different default contructors; the construcotr's job is to create objects themselves, and not copy things between two EXISTING objects. So we don't expect anything to get printed out here!

	return 0;
}

```
- We're not doing any construction here, just copying; because it's just assignment and not construction, because myCube was already constructed here.
- Because it's assignment and not construction... we expect he program to output default constructor twice and nothing else

--

So we've said that classes have constructors; ways to initialize the class when created.

The ==copy assignment operator== defines the behavior when an object is copied using the assignment operator `=`

![[Pasted image 20231216212558.png]]

Up until now, we've talked about classes having constructors; ways to initialize values of the class when it's created. Now we want to pivot slightly and talk about the copy assignment operator, which defines the behavior of using the assignment operator `=` in code.

Let's see how this works!

A ==copy constructor== *creates a new object* (constructor)
An ==assignment operator== replaces the value of an existing object.
- Every object has to be constructed; built by a constructor. Once it's been constructed, it can't be constructed again.
- To change the value of an *existing class*, one that's already been construceted, it must be changed through an assignment operation.

C++ provides a default ==automatic assignment operator== if you don't define your own; it will do everything that you probably need it to do; Only when when have externally-allocated resources (eg memory) or if we want multiple objects to point to the same thing, do we need to create a ==custom assignment operator==

==Custom assignment operators== must:
1. Be a public member function of the class
2. Has the function name `operater=` (a very specific name)
3. Has a return value of a reference of the class' type (the return value must be a Cube by reference, if it's a Cube)
4. Has exactly one argument (this argument must be a *const reference* of the class' type)
	- `Cube & Cube::operator=(const Cube & obj)`
	- The one argument is a const reference to the class's type . The & is "by reference," and the obj is object.
	- The goal is to assign the contents in object (obj) to the instance of the class that it's being called upon (Cube)


Let's look at an example:

In an implementation file
```cpp
...

namespace uiuc {
	Cube::Cube() { // A custom default constructor
		length_ = 1;
		std::cout << "Default custom constructor invoked" << std::endl;
	}

	Cube::Cube(const Cube & obj) {
		// This is our custom copy constructor, where we're taking the values from obj and we're setting them on our new Cube object
		length_ = obj.length_;
		std::cout << "Copy constructor invoked!" << std::endl;
	}

	Cube & Cube::operator=(const Cube & obj) {
		// This is our custom assignment operator, where we already have a cube object and we want to copy some stuff from obj into/over it.
		// The "Cube &" means that it returns a cube by reference. Its function name is "operator=", and hte one parameter it has is a "const Cube & obj", a "constant cube by reference"
		length_ = obj.length_;
		std::cout << "Assignment operator invoked!" << std::endl;
		// We always return a dereferenced value of this; this is an instance of the class itself, so we'rej just returning an instance of the class
		return *this; 
	}
}
```

Applications of this code

```cpp
#include "Cube.h"
using uiuc::Cube;

int main() {
	Cube c; // Invokes a default constructor
	Cube myCube; // Invokes a default constructor

	// Invokes our custom assignment operator; Both of these objects already existed, so we are invoking the assignment operation because both objects already exist...rather than the copy constructor
	myCube = c;

	return 0;
}

```
The invocation of the assignment operator means that both objects already exist and don't need to be constructed.

Let's learn more about the construction of a C++ class!
![[Pasted image 20231218144450.png]]
- `Cube *ptr = &c` is saying "Create a pointer to a cube object whose value is the memory address of the Cube c"
- `Cube &r = c`  is creating *reference variable* for c.

When an instance of a variable (eg a Cube class) can be stored in 3 ways:
- Stored in memory directly
- Accessed via pointer
- Accessed via reference


*Direct Storage*
- By default, variables are stored *directly* in memory.
	- The `type` of a variable has no modifiers
	- The object takes up *exactly* its size in memory.
```cpp
Cube c; // Stores a Cube in stack memory
int i; // Stores an integer in memory
uiuc::HSLAPixel p; // Stores a pixel in memory
```

*Storage by Pointer*
- The *type* of a variable is modified with an asterisk (`*`)
- A pointer takes a "memory address width" of memory (ex: 64 bits on a 64-bit system)
- The pointer "points" to the memory that's allocated for the object (the allocated space of the object)
```cpp
Cube *c; // Pointer to a Cube in memory
int *i; // Pointer to a int in memory
uiuc::HSLAPixel *p; // Pointer to a pixel in memory
```

*Storage by Reference*
- A reference is an ABLIAS to existing memory, and is denoted in the type with an ampersand (`&`)
- A reference *does not store memory itself*... it is only an alias to another variable.
- The alias must be assigned when the variable is initialized!
```cpp
Cube &c = cube;  // An alias to the variable `cube`
int &i = count; // An alias to the variable `count`
uiuc::HSLAPixel &p; // !! ILLEGAL! You MUST alias something when these reference variables are initialized! Won't even compile.
```


Let's look at why this matters in an example!
![[Pasted image 20231218145310.png]]
Say we have our Cube class, and we want to treat our cubes as if they're currency. The volume of the cube is the $value of the cube.

Whenever we receive money, we want to make sure that we're getting the cube itself -- not a copy of a cube. We want to actually transfer the cube to someone else; Creating a copy of a cube would create inflation, we'd be a bad banker!

Modifying the cube class so we have some output so that we have context on what the Cube is doing:
Cube.cpp
```cpp
...

// Notice the (returned?) classname is Cube, same as the class itself... it has one parameter, no return value, so it has to be a constructor. This is a one parameter constructor.
Cube::Cube(double length) {
	// A custom, one-parameter constructor
	length_ = length;
	std::cout << "Created $" << getVolume() << std::endl;
}

// The copy constructor similarly returns a Cube. It takes in a Cube by reference (see the &?). This means that it's not going to create a new cube; we're simply going to have the cube aliased to us... This aliased cube will be used to create a new Cube.
Cube::Cube(const Cube & obj) {
	// This is a copy constructor, where we're given an "obj" cube and asked to make a new one
	length_ = obj.length;
	std::cout << "Created $" << getVolume() << " via copy" << std::endl;
}

Cube & Cube::operator=(const Cube & obj) {
	// This is our custom assignment operator. This transforms an instance from one value to another value.
	// We're going to say "Transformed $20 -> $20", or something like that. We're going to print the value it has before to the value that it's going to have afterwards.
	std::cout << "Transformed $" << getVolume() << "-> $" << obj.getVolume() << std::endl;
	length_ = obj.length_
	return *this;
}
```

All three of these functions are concerning because all things are "creating money". By analogy, when an object is created, it's taking up memory! There are resources that are allocated when these operations are performed.

Let's see an example of this being used:
```cpp

// What happens when we transfer a cube to another cube by value?
int main() {
	// Create a $1,000-valued cube
	Cube c(10);

	// Transfer the cube. This is going to invoke the copy constructor, becuase myCube didn't exist.
	Cube myCube = c;

	return 0;
}
```
If we ran this, we'd see:
```
Created $1000
Created $1000 via copy
```

So we created some extra $ (memory) here. Let's see if using a different memory technique would help us out

```cpp
int main() {
	// Creat a $1000 Cube using our custom constructor
	Cube c(10);

	// Transfer the $ to another person by ALIASING the money, because we have a reference variable
	// myCube becomes an _alias_ of c.
	Cube & myCube = c;
	// So both myCube and c are both... the same cube. We expect the cube to have been constructed exactly once.
}
```
If we ran it,
```
Created $1000
```
Just created a single cube worth $1,000!
We haven't really transferred anything though. We've just aliased two people to owning the same cube.

Let's try something else:
```cpp
int main() {
	// Create a $1,000-valued cube
	Cube c(10);

	// Transfe the cube
	// Instead of having an alias, where we have two variables that are the exact same cube
	// here we have two variables; one is the cube, and the other is a pointer to the cube.
	Cube *myCube = &c;

	return 0;
}
```
If we ran it,
```
Created $1000
```
This makes sense

Let's try to actually "transfer" some money, though.
- So far, we've been doing everything in the main function. What we really care about is passing money around so that we can have a function like transferMoney, so that we can give money to someone else.
- There are 3 different ways of doing this (similar to the 3 different ways of storing access to variables):
	- Pass by value (default)
	- Pass by pointer (modified with *)
	- Pass by reference (modified with &, acts as an alias)

Let's look at an exmaple:
```cpp

bool sendCube(Cube c) {
	// logic to send a cube somewhere
	return true;
}

int main() {
	// Create a $1,000-valued cube
	Cube c(10);

	// Send the cube to someone
	// Because it's being sent by value, it needs to be copied from the main function stack memory to the sendCube stack memory.
	// So we expect to see the copy constructor invoked!
	sendCube(c);

	return 0;
}

```
We see:
```
Created $1000
Created $1000 via copy
```
- I'm preeetty sure that we don't see another copy because we aren't assigning the returned object to anything, in the main function.
- So we're really sending someone a copy of our cube, and not the cube itself.

Let's try something else:
```cpp

// See this ampersand? It means that we're sending the variable by reference. This means that we aren't copying the variable anywhere; we're just sending an alias for the varibale.
bool sendCube(Cube & c) {
	// ...logic to send cube
	return true;
}

int main() {
	// Create our $1,000 cube
	Cube c(10);

	// Send the cube to someone
	// Note that we don't have to invoke this any differently
	sendCube(c);

	return 0;
}

```
We expect this to only be created once and no copies created.
```
Created $1000
```
Cool!
The takeaway there is that we didn't have to invoke that any differently from the first example...


Let's try it by pointer
```cpp

bool sendCube (Cube * c) {
	return true;
}

int main(){
	// Create our $1,000
	Cube c(10);

	// Transfer our cube by sending the memory address of the dang cube
	sendCube(&c);

	return 0;
}
```
Results in 
```
Created $1000
```
So we didn't have to create a second cube because we just passed a pointer to our cube.

The final thing to mention:
- We can return by *all of these different types* (value, pointer, reference).
![[Pasted image 20231218152859.png]]
- We shouldn't do the stuff in red because then that actual memory is reclaimed when that function exits as the stack memory of the function is cleaned up.


The big idea to ==take away==:
- We have three ways of storing variables (directly, by pointer, or alias to some other existing memory)
- These three different ways of storing data allow us to be very flexible and control exactly how our memory is being copied around.
- The Cube class provides a great framework to play around with the code and see how the cube is copied/created as you do things with it.


### Video: Class Destructor
Let's now talk about ==Class Destructors!==
- When an instance of a class is cleaned up, the class destructor is the last call in a class's lifecycle!
![[Pasted image 20231218153547.png]]

Just like all the other topics we've talked about recently, there's an ==automatic default destructor== provided for you *for free* as long as you have no other destructor defined for your class.
- The only action of the automatic default destructor is to call the default destructor of all member objects
	- If you want to do things like logging or cleaning up other memory, you'll have to write your own destructor.

Destructors should NEVER be called directly!
- The destructors are called during runtime; the compiler will insert implicit, automatic calls to the destructor in appropriate places where stack objects should be destroyed as they go out of scope, or where heap objects are destroyed using `delete`. This is why you don't need to call the destructor explicitly yourself.

Instead, the destructors are *automatically called* when the object's memory is being reclaimed by the system!
- If the object is on the STACK, destructors are called when the function returns.
- If the objet is on the HEAP, destructors are called when `delete` is used.

Writing ==Custom Destructors==
- To add a Custom Destructor to a class so that we have control over what happens at the end of the class lifecycle, we need to create a function that's defined as:
	- A custom destructor is a member function
	- The function's destructor is the name of the class itself, preceded by a `~`
	- All destructors have zero arguments, and no return type. 
		- There's no ability to change up how we define a custom constructor, this is it :) 
```cpp
Cube::~Cube();  // A Custom destructor
```


To see how this works, let's continue our currency example we saw in the previous video:
```cpp
#include "Cube.h"
#include <iostream>

using std::cout;


Cube::Cube() {
	// Our custom default constructor
	// This is going to create a cube having length_ of 1, iirc. Which is a $1 cube.
	cout << "Created $1 (default)" << endl;
}

Cube::Cube(double length) {
	// Another custom constructor, this one taking one argument of the length
	// This prints out the $ of the cube, defined by the volume that's determined by the passed length 
	// Sam: I think we're omitting some bits here where we'd acutally set length_=length;
	cout << "Created $" << getVolume() << endl;
}

Cube::Cube(const Cube & obj) {
	// A copy constructor where we're being passed a reference to an existing Cube object `obj`
	// Sam: There might actually be some code here in reality to copy stuff over.
	cout << "Created $" << getVolume() << " via copy" << endl;
}

Cube::~Cube() {
	// Custom destructor
	cout << "Destroyed $" << getVolume() << endl;
}

Cube & Cube::operater=(const Cube & obj) {
	// A custom assignment operator. Given that we already have a cube, and we're doing cubeA = cubeB, here cubeB is this obj object being passed as an argument. This is going to overwrite some of our own stuff, using that obj's stuff.
	cout << "Transformed $" << getVolume() << "-> $" << obj.getVolume() << endl;
}

```

So now every operation is going to print out a slightly different line of code.
Now we can work through a program and see what's actually happening.
Let's look at some code:

```cpp

double cube_on_stack() {
	// Create a cube using our custom constructor. Thsi cube is valued at $27. Nice.
	// This cube lives in the stack for this function
	Cube c(3);
	// we return an integer, which also lives in the stack for this function.
	return c.getVolume()
}

void cube_on_heap() {
	// Create a cube with value $1,000 on the heap, and have a pointer c1 that points to it in stack memory.
	Cube * c1 = new Cube(10); 
	// Create a unit cube ($1) on the heap, and have a pointer c2 that points to it in stack memory
	Cube * c2 = new Cube;
	// Delete the cube object on the heap memory that c1 points to.
	delete c1;
}


int main() {
	cube_on_stack();
	cube_on_heap();
	cube_on_stack();
	return 0;
}

```

```
Created $27 (as we create c(3) in cube_on_stack)
Destroyed $27 (as we destroy the cube in the stack memory of cube_on_stack as it returns)
Created $1000 (As we create the c1-pointed cube in heap memory in cube_on_heap)
Created $1 (As we created the c2-pointed cube in heap memory in cube_on_heap)
Destroyed $1000 (As we explicitly called delete on the $1000 in heap memory in cube_on_heap)

( Because c2 is in heap memory, we don't destroy c2! Because it's in heap memory, we don't destory it when the function returns!)

Created $27 (from the second cal to cube_on_stack) 
Destroyed $27 (from the second cal to cube_on_stack)

```


The big takeaway with custom destructors
- They're necessary when we have to close/free memory or objects associated with some object in question
	- If we created some new memory inside the class on the heap, we need to destroy it
	- If we opened other files 
	- or shared other memory

Those are the usages we see in custom constructors! With a custom constructor, we have a great idea of the entire lifecycle of the class, from the constructors, to assignment/copy operations, to the end-of-life of our class with destructors.

So how do we build awesome things in C++? More on that later!

------

### Reading: C++ Syntax Notes: Uninitialized Pointers, Segfaults, and Undefined behavior

==Segmentation fault (Segfault)==
- Various types of programming bugs related to pointers and memory can cause your program to crash; in Linux, when you dereference an address that you shouldn't, this is often called a "segmentation fault".
	- For example if you dereference a pointer that is set to `nullptr`, it will almost certainly segfault and crash immediately:
```cpp
// Bad code that will segfault because of an attempt to dereference teh nullpointer
int* n = nullptr;
std::cout << *n << std::endl; // Boom
```
==ABOVE, NOTE:==
- In c++, these are all syntactically correct and functionally equivalent:
	- `int* n = nullptr;` (type-focused style)
	- `int * n = nullptr;`  (balanced style)
	- `int *n = nullptr;` (variable-focused style)
- The choice between these styles is largely a matter of personal or team preferences, and different codebases may adopt different conventions for readability or consistency.

==Undefined behavior==
- Sometimes it *is* possible to write a program that compiles, but that isn't really a safe or valid program.
- Improper ways to write C++ that aren't strictly forbidden by the C++ standard are called undefined behavior.
- Many beginning programmers make the mistake of thinking that just because their program compiles and runs, that it must be safe and valid code -- THIS IS NOT TRUE! YOU MIGHT STILL HVAE UNDEFINED BEHAVIOR!
	- Proofread your code, use safe practices, and avoid relying on undefined behavior.
	- Many times, undefined behavior is caused by the careless use of uninitialized variables. Let's talk about it in the next section

==Initialization==
- Initialization is specify a *value* for a variable from the *moment that it's created!*
```cpp
int* x; // This is dangerous; it can lead to careless mistakes and crashes!
int* y = nullptr; // This is explicitly initializing a pointer to nullptr
```
- The pointer "x" above is uninitialized; it contains a seemingly random memory address -- dereferencing this pointer would cause undefined (unpredictable) behavior!
- In contrast, for pointer "y" above, we *do* initialize it -- but we've initialized it to `nullptr` -- dereferencing this pointer would cause the program to crash immediately and predictably.

You can also initialize a value with the () syntax following the variable name:
```cpp
int* y2(nullptr); // ()
int* y3{nullptr}; // {} is also fine
```
- If the type is a class, the parameters will be given to the class type's corresponding constructor. For built-in types like int, which aren't class types, there are no constructors; however we can still specify an initialization value this way.
	- Sometimes you'll see this initialization done with the {} syntax instead of () syntax -- this is a new feature since C++11, and can be a good way to make it clear that you'er performing an initialization, not some kind of function call -- later in this course sequence, we'll see some other good ways to use thsi {} syntax.

Plain built-in types like `int` that are not initialized with have *unknown* values! 
However if you have a class type, its default constructor will be used to initialize that object.
```cpp
// h is an uninitialized built-in type
int h;
// b will be default-initialized by the default constructor of the Box class (the automatic default contructor if it doesn't have a custom one defined)
Box b;
```

Below, we're going to create an integer "i" on the stack, safely initialized with the value 7, and then create a pointer "z" on the stack, initialized to the *address* of i.
Because z points to i (which has an initialized value), it's safe to dereference z.
```cpp
int i = 7;
int* z = &i;
```

##### Examples with Heap memory...

If we wanted to use heap memory, we'll use the `new` operator, like we learned in class.
- Be way when we use "new" for a built-in type like int, since these types may not have any default initialization! 
	- Therefore, you shouldn't assume that they'll have any expected default value (such as 0). For those cases, it's best to initialize the value manually! Here's some examples:

```cpp
// This is us creating and initializing a q pointer with the address for a newly allocated integer on the heap
// Recall that new ... returns the memory address of the start of the object in the heap.
// Because we didn't initalize this integer with any predictable value, we shuldn't rely on this integer to have any particular value.
int* q = new int;

// You can specify initialization parameters at the end of teh "new" expression:
int* r = new int(0);
```

There are a lot of special situations in C++ where different factors might slightly change how an object is initialized. We don't need to get into all the details; if you're unsure, the best thing to do is to explicitly initialize your variables!


##### Resetting deleted pointers to nullptr

Now that we've reviewed initialization, especially for pointers, let's talk about why ==we should manually reset pointers to nullptr when we're done with them!==

Note that using "delete" on a pointer frees the heap memory allocated at that address. However, deleting the pointer does NOT change the *pointer value* itself to "nullptr" automatically! 
- You should do this *yourself* after using `delete`, as a safety precaution:
```cpp
// Allocate some integer on the stack
int* x = new int; // Not initialized with a value; Doesn't matter for this demonstration, but isn't the best practice, probably.

// Now we know that x holds some memory address that points to a valid integer.
// Let's do some kind of work with that integer that our x points to by assigning to that pointer's dereferenced value.
*x = 7;

// Now, let's use the delete keyword to deallocate the heap memory!
delete x;

// This destroy/de-alllocates the integer on the heap and frees the memory, but now x still holds that memory address!
// We should set x's value to the nullptr for safety (recall that you can think nullptr is a keywork that is a pointer to some memory location like 0x0)
x = nullptr;
```
The idea is that by setting x to nullptr explicitly after deleting the dereference value of x, we avoid two problems:
1. We don't want to delete the same allocated address more than once by mistake (eg later running `delete x` again, which could cause errors. 
	- Using `delete x` again when `x=nullptr` does nothing, so no further error happens.
2. We must never dereference a pointer to deallocated memory! 
	- This could cause the program to crash with a segfault, exhibit strange behavior, or cause some security vulnerability -- this variance in outcome isn't a great thing.
	- Attempting to dereference a `nullptr` will almost always just cause the program to segfault/terminate immediately! This well-defined "bad" behavior is much better than the not-well-defined "bad" behavior above -- so it makes sense to allocate the deleted pointer to nullptr, thus ensuring that *if we dereferenced it carelessly*, then it will cause a very obvious runtime error that we can fix!

We should note that we only need to use `delete` and `nullptr` with pointers! 
- In contrast, simple variables in stack memory don't need to be manually deleted; These are automatically deallocated when they go out of scope.

In ==summary==: Remember that if you use `new`, you should also need to use `delete`, and after you `delete`, you should set any relevant pointers to `nullptr`.

Growing beyond pointers:
- As we go further in lessons, we might be frustrated that messing with pointers and raw memory is very tedious -- However, class types can be designed to handle all the new and delete operations for you -- invisibly.
- As you create your own robust data structures, and as you use libraries like the C++ Standard Template Library, you'll find that we very rarely have to even use `new` and `delete` anymore.

----

### C++ Syntax Notes: The Modern Range-based "for" loop

In recent versions of C++, there's a versions of the `for` loop that automatically iterates over all of the things in a container.
This is very useful when used with a standard library container, because you don't have to worry about accidentally accessing memory outside of a safe range.

```cpp
// Recall the origianl for loop
for (declaration; condition; incrementOperation) { loop body } // The original for loop

// Here's teh NEW one
for (temporary variable declartion: container) { loop body } // The "newer" shorthand that you can use
```

There's an important detail about the temporary variable, in the new. example If you declare an ordinary temporary variable in the loop, it just gets a *copy* of the current loop item by value...so changes you make to it don't affect the actual container!

```cpp
#include <iostream>
#include <vector>

int main() {
	// In the standard library, std::vector is an array with an AUTOMATIC size!
	// Let's make a vector of ints and loop over the contents
	// The syntax for std::vector<> is discussed further in the lecture on template types.

	// We declare (but don't initialize)
	std::vector<int> int_list;
	int_list.push_back(1); // push_back appends the given element `value` to the end of the vector container
	int_list.push_back(2);
	int_list.push_back(3);

	// Automatically lop over each item, one at a time, using the C++11 (?) syntax:
	for (int x : int_list) {
		// Recall: For {temporary variable} in {container}, baiscally
		// This makes a temporary copy of each element in the list by value; So if we mutate the element during our iteration, we won't actually be making any mutation on the elements that are in the container.
		x = 99;
	}
	// It prints as 1,2,3
	// Above: We looped over the int_list container (length 3) and tried to use the assignment operator to set each one to 99, but because the iteration is creating a copy of each element by value, we don't actually end up making any changes:

	// Looking back over the same container, we'll see that none of the elements in the container were mutated
	for (int x: int_list) {
		std::cout << "This item has value: " << x << std::endl;
	}

	return 0;
}
```

If we instead make the temporary variable of a *reference type*, then we CAN actually modify the current container item! 

```cpp
#include <iostream>
#include <vector>

int main() {
	std::vector<int> int_list; // Declare but don't instntiate a vector-typed int_list variable.
	int_list.push_back(1); // Append some objects to this motherfucker
	int_list.push_back(2);
	int_list.push_back(3);

	// Let's do an iteration over the container like list time, but instead of doing `int x`, let's do `int& x`, where we're 
	for (int& x : int_list) {
		// This version of the loop will actually get a reference variable to each of the elements that er in our container
		x = 99;
	}

	// Now when we actually iterate over the container again and print the values, we're going to see that the values have been changed:
	for (int x: int_list) {
		std::cout << "Value: " << x << std::endln;
	}
	// It prints as 99, 99, 99
}

```

There are some more advanced ways to use this too!
- Let's say that we're iterating over some large objects in a container.
	- Even if we DON'T want to modify the objects, we might actually want to use a *reference to a constant* as the loop variable type to avoid making a temporary copy of a large object (as we would without the &), which could be slow and resource-intensive.

```cpp
#include <iostream>
#include <vector>

int main() {
	std::vector<int> int_list;
	int_list.push_back(1);
	int_list.push_back(2);
	int_list.push_back(3);

	// See that, relative to our last example, we've prepending with a "const"
	for (const int& x : int_list) {
		// If we tried to modify 
		std::cout << "Value: ", x << std::endl;
	}

	return 0;
}

```

---
Aside:
- The `const` keyword in C++ is used to define variables whose value cannot be changed!
	- When used in a reference in range-based for loops like above...
		- In the `for (int& x : int_list) {}` case
			- We get a reference to each of the actual elements in the list. Since it's a reference, we can modify the actual elements of int_list within a loop!
		- In the `for (const int& x : int_list) {}` case
			- In this case, the elements are accessed as *constant references*. This means that you can't modify the elements of int_list through these references.
			- This is especially useful when you want to iterate over a container to read the elements without the risk of modifying them.

So:
- Normal for loop
	- Creates a copy of each element in memory ($) and you can't really modify them
- Reference for loop
	- Does not create a copy of each element in memory, and you can modify them
- Const Reference for loop
	- Doesn't create a copy of each element in memory, and you can't modify them

---

#### Unsigned Integer Types: Be Careful

- ==Unsigned integers== are an integer type that can't represent negative values.
	- Instead, unsigned integers have an increased *upper positive value range* compared to signed integers of the same memory usage (signed integers devote a bit of memory just to even be able to represent a negative sign).
	- Actually *mixing* both signed and unsigned integers in your code can cause unexpected problem!

Let's look at some examples first:

```cpp
int a = 7; // The normal "int" syntax creates a signed int by default

unsigned int b = 8;  // If you write either of these, you create an unsigned integer type
unsigned c = 9;
```

Issues with unsigned arithmetic
```cpp
std::cout << (a+b) << std::endl;
```
- Addition with unsigned integers might not be a problem, as long as you don't exceed the maximum range. This is 7+8 and shows 15 as expected!

But you need to be careful when we're approaching the upper limit for a signed int, even when we're using unsigned ints!
- Signed ints have a relatively lower maximum value, so if you get an unsigned int *greater* than the limit of a signed it, and then try to cast it to a signed it, it will be interpreted as some unexpected negative number!

There are some issues that can arise with subtraction as well:
```cpp
unsigned int x = 10;
unsigned int y = 20;

std::Cout << (y-x) << std::endl; // 10
std::Cout << (x-y) << std::endl; // !!! This outputs 4294967286
```
- If you try to imply negative values using unsigned ints, this instead results in a very large positive number! In this situation, the output is 4294967286, which is close to the maximum for an unsigned 32-bit number.


Perhaps strangely, if you explicitly case the result back to a signed integer, then we get something usable again! The outcome of this same subtraction of two unsigned integers (which previously yielded a large positive number), when cast to a signed integer, results in a -10, which is what we'd expect!
```cpp
std::cout << (int)(x-y) << std::endl; // -10
```

You can also do a casting operation to convert from unsigned to signed int by assigning some unsigned result to a signed int variable!
```cpp
int z = x - y;  // Here, z is a signed-int type, and x and y are both unsigned ints. The result of the subtraction of x and y is some large positive number, but the (implicit) conversion of it back to a signed integer results in the "correct" -10.
```

Making direct comparisons between signed and unsigned ints can also cause issues! This next line may give a warning or error. We've commented it out for now:
```cpp
std::cout << (a<b) << std::endl; // This comparison between a signed and unsigned integer may cause a warning or error!
```


==Factoid: Container Sizes are often unsigned!==
- We often refer to generic data structure classes as "containers." The ==C++ Standard Library Template (STL)== provides many of these, such as std::vector.
- Many of these class types have a `size()` member variable that returns the number of items that the container currently holds.
	- It's important to note that in many cases, this actually gives you the size in an UNSIGNED integer type!
		- This makes sense, because the cardinality/size of some container isn't going to be a negative number.
		-  The DOWNSIDE of this is that sometimes you'll accidentally run into cases where you're comparing a *signed int* (perhaps an iteration counter) with an unsigned integer size!
			- So be prepared!

```cpp
std::vector<int> v = {1,2,3,4};

// At each loop here, we're going to be comparing our signed integer `i` to some unsigned integer that's returned from `v.size()`
// The compiler is going to warn us that it's comparing signed and unsigned integers!
for (int i =0; i < v.size(); i++) {
	std::cout << v[i] << std::endl;
}
```

You could handle this by using various casting methods to make the warning go away, or you could simply use an unsigned int, or indeed `std::vector<int>::size_type` itself as the type for the counter i.


Let's consider the danger of trying to *subtract* from the unsigned int that represents the size!
```cpp
std::vector<int> v = {1,2,3,4};

for (int i = 0; i <= v.size()-1; i++) {
	std::cout << v[i] << std::endl;
}

// What might go wrong, above? What if v.size() were 0? Then the program would crash as it tries to run a very large number of loops, and ends up doing a (python IndexError), accessing incorrect memory and segfaulting/crashing.

```

Here's some other tricks to solve the problem:
```cpp
// Casting to signed int first helps to ensure that the result of subtraction will truly be a signed negative value when the size is 0
for (int i = 0; (int)v.size()-1; i++) {
	// ...
}

// Rewriting the algebra to perform ADDITION, instead of subtraction
for (int i =0; i+1 <= v.size(); i++) {
	// ...
}
```


In conclusion, be careful when you're dealing with unsigned integers!
- For everyday high-level programming purposes, *signed integers* may be all that you need!
- If you need to ensure positive values during your code execution, you can write safety checks into the code to monitor values and issue warnings or errors if a certain range is exhibited, rather than using unsigned integers.


-----

## Week 3 Quiz

![[Pasted image 20231218182254.png]]
- The answer is actually C.
	- Classes can have multiple constructors defined for it
	- If a constructor isn't declared, an automatic default constructor will be defined for it by your C++ compiler.
	- Constructors don't have a return type in C++


Re: #2... Do these call the copy constructor?

```cpp
// Function prototype for "intersect":
Cube intersect(Cube &left, Cube &right);
// ...
Cube a(10),b(5);
Cube c;
c = intersect(a,b);
```
- In this case, we're passing the a and b cubes to the intersect function... but the function itself has the types defined as reference variables, so they aren't copied into the stack memory for the function. The fact that the function returns a Cube, rather than a reference or a pointer, suggests that the function is returning a new Cube object by value, which would require copying something back into the main function's memory when we do `c = intersect(a, b);`

```cpp
Cube b(10);
Cube a = b;
```
- This is clearly going to call the copy constructor, because a isn't yet initialized, but b is.

```cpp
Cube a,b(10);
a = b;
```
- This is NOT going to call the copy constructor. Both a and b have been declared as Cube variables, so a = b is going to use the (perhaps custom) assignment operator that's defined for the Cube class.

```cpp
// Function prototype for "contains":
int contains(Cube outer, Cube inner);
// ...
Cube a(10),b(5);
int a_bounds_b = contains(a,b);
```
- In this case, the contains function doesn't use * or & in the parameter definitions, so arguments that are passed to it on invocation have to be copied into the stack memory of the function.

Question #3
![[Pasted image 20231218183049.png]]

Recall Custom assignment operators must:
1. Be a public member function of the class
2. Has the function name `operater=` (a very specific name)
3. Has a return value of a reference of the class' type (the return value must be a Cube by reference, if it's a Cube)
4. Has exactly one argument (this argument must be a *const reference* of the class' type)
	- `Cube& Cube::operator=(const Cube& obj)`
	- The one argument is a const reference to the class's type . The & is "by reference," and the obj is object.
	- The goal is to assign the contents in object (obj) to the instance of the class that it's being called upon (Cube)

So to answer the question:
- The return type of the custom assignment custom assignment operator is the class' type itself
- The custom assignment operator is a function declared with the name (eg) Cube::operator=(...)
- The custom assignment operator function is declared with ONE argument -- The source object (rather, a const reference to the source object)


Question #4
```cpp
class Orange {
  public:
    Orange(double weight);
    ~Orange();
    double getWeight();
  
  private:
    double weight_;
};

```
- Select ALL functions that are present in this class (INCLUDING any automatic/implicit functions added by the compiler):
	- Default constructor is NOT present, and there won't be any automatic default constructor created, because we've defined an custom non-default constructor on the line below.
	- Custom, Non-default constructors are present in the Orange(double weight); bit
	- Copy constructor is NOT present, but it is automatically added
	- Assignment operator is NOT present, but it is automatically addded 
	- Destructor IS present! The `~Orange()` defines the destructor of the class.


Consider the following class:
```cpp

class Blue {
  public:
    double getValue();
    void setValue(double value);

  private:
    double value_;
};

```
- Select the functions that re present in this class (INCLUDING automatic/implicit functions added by the compiler!)
	- Default Constrcutor isn't explicitly provided, but there will be a automatic default one created
	- There's no custom non-default constructor defined
	- Copy constructor is automatically defined for us
	- Assignment operator is automatically defined for us
	- Destructor is automatically defined for us.

```cpp

class Animal {
  public:
    Animal(); // custom default
    Animal(std::string name); // custom non default
    Animal(std::string name, int age); // custom non default
    Animal(std::string name, int age, double weight); // custom non default
    
    Animal(const Animal& other); // This is a copy constructor, but that counts as a constructor
    
    void setName(std::string name); // Just a fn
    std::string getName(); // Just a fn

  private:
    // ...
};

```
- How many EXPLICIT (non-automatic) constructors are there?
	- 5

Question #7: 
When you use the **new** operator to create a class object instance in heap memory, the **new** operator makes sure that memory is allocated in the heap for the object, and then it initializes the object instance by automatically calling the class constructor. After a class object instance has been created in heap memory with new, when is the destructor usually called?
- If there's been some memory allocated in the heap for the object, it's up to the programmer to manually delete/free that memory using the `delete` operator. So you never call the destructor function directly, manually. Instead, the instructor is invoked for you when the delete operator is used. If this were memory of something that were on the stack, THEN it would have been called automatically when the variables goes out of scope.


Question #8
```cpp
double magic(uiuc::Cube cube) {
  cube.setLength(1);
  return cube.getVolume();
}

int main() {
  uiuc::Cube c(10);
  magic(c);
  return 0;
}
```
- How many times is the uiuc::Cube's copy constructor invoked?
	- Once to copy the Cube c in main stack memory into the function stack memory.
(We're just returning an int from the function, and it isn't being assigned anywhere in the main memory, so there isn't a second invocation)

Question #9
We've looked at examples where the assignment operator returns the value `*this`. The variable `this` is available by default in most class member functions. What's the value of the built-in class variable `this`?
- `this` is a built-in variable that's a pointer that holds the address of the current object!
	- It's automatically available in all non-static member functions of a class.
- It's a self-referential pointer used in class member functions in C++ to point to the object for which the function is being invoked, allowing access to the object's members and enabling method chaining and disambiguation in certain scenarios.

So the answer her is
- A pointer to the current object instance

Question #10
- Consider the code below that includes a class that has a custom constructor and dstructor, and both utilize a global variable (which has global scope and can be accessed anywhere and is initialized before the function main is executed.)
```cpp
int reference_count = 0;

class Track {
	public:
		Track() { reference_counter++; }
		~Track() { reference_count--; }
};
```
Which one of the following procedures (void functions) properly ensures the deallocation of ALL the memory allocated for objects of type Track, so that the memory can be re-used for something else after the procedure runs?

```cpp
void track_stuff() {
    Track *t = new Track; // Creates a new Track object in heap memory, and a poitner to it in stack memory.
    // ...
    // Recall: When an object is stored via a pointer, access can be made to member functions using the -> operator!
    t->~Track(); 
    return;
}
```
- This is us explicitly calling the destructor. We shouldn't be doing that, probably...
	- This will result in the `reference_count` variable being decremented TWICE; once in the explicit destructor call, and once when our t variable goes out of scope.

```cpp
void track_stuff() {
    Track t;
    // ...
    delete t;
    return;
}
```
- This is us invoking the destructor using the `delete` operator, which is the appropriate way to do it... HOWEVER we're using the delete operator on a stack-allocated Track object. This should cause a syntax error of some sort. The delete operator should be used to deallocate memory for objects that were allocated on the heap using the `new` keyword.

```cpp
void track_stuff() {
    Track t;
    Track *p = new Track;
    // ...
    delete p;
    return;
}
```
* ✅ This is us creating a Track in stack memory, creating ANOTHER Track in heap memory with a pointer in stack memory, and then deleting the Track that the pointer p points to. This still leaves t in stack memory, but that is automatically deallocated when the function returns, so this is a good answer!

```cpp
void track_stuff() {
    Track t;
    Track *p = &t;
    // ...
    delete p;
    return;
}
```
- This is us creating a Track in stack memory, and a pointer in stack memory that points to the address of that track in stack memory. This use of the `delete` operator is undefined behavior is C++. The use of the `delete` operator on an object that wasn't allocated using `new` (it's in the stack memory) is undefined behavior that might crash the program, cause a memory leak, or lead to other unpredictable behavior.

---
# Programming Assignment

A class called Pair has been declared, but the constructors haven't been implemented yet.
- Pair has two public member variables
	- int* pa 
	- int* pb
- These two "pointers to int" are intended to point to heap memory locations that store integers. The remainder of the Pair class expects the following functionality:
	1. A single constructor `Pair(int a, int b)`. This should set up pa and pb to point to newly allocated memory regions on the heap. The integers at those memory locations should be assigned values according to the constructor's integer arguments a and b.
	2. A copy constructor `Pair(const Pair& other)` : this takes as its argument a read-only reference to another Pair. It should set up the newly constructed Pair as a "deep copy." That is, it should create a new Pair that is equivalent to the other Pair based on dereferenced values, but doesn't reuse any of the same memory locations. 
		- In other words, the copy constructor should set up its own instance's member variables pa and pb to point to newly allocated memory locations for integers on the heap; those memory locations must be new, not the same locations pointed to by the other Pair.
	3. A destructor `~Pair()` that de-allocates all of the heap memory that had previously been allocated for this Pair's members.













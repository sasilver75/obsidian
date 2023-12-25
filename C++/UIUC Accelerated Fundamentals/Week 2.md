
Our focus this week is the C++ memory model! As part of this, we're going to see:
- Stack memory
- Heap memory
- References
- Pointers

Let's get into it 💪

-----

C++ Programs have control over the memory and lifecycle of every variable!

By default, variables live in ==stack memory==

Every variable have four things:
- Name
- Type
- Value
- Location in memory ("memory address")

```cpp

int primeNumber = 7;
```
- The type is int, the name is primeNumber, and the value is 7. The memory address isn't known, but we can explore where it is using the & operator
- The & operator returns the memory address of a variable!
```cpp
#include <iostream>
int main() {
	int num = 7;

	std::cout << "Value: " << num << std::endl;
	std::cout << "Address: " << &num << std::endl;

	return 0
}
```
- We're using `&num` to ask the cpp language to give us the address where that memory is being stored.
- Because this is a ==stack== memory address, this address is going to be a somewhat large number
	- eg. 0xffff27242
		- This is written in hexadecimal (base-16)

This very high memory address is exactly what we'd expect of stack memory.
Let 's look at WHY this is a high memory address, and understand how memory works on our computer!

==Stack Memory== is the DEFAULT place where every variables in C++ is placed by default!
- Stack memory is associated with the CURRENT FUNCTION; and the memory's lifecycle is tied to the function!
- When the function returns or ends, the stack memory of that function is released.
- Stack memory STARTS at high addresses and GROWNS DOWN towards zero!

Let's see this happen when we have two diff functions

```cpp
#incldue <iostream>

void foo {
	int x = 42;
	std::cout << " x in foo: " << x << std:endl;
	std::cout << "&x in foo: " << &x << std:endl;
}

int main() {
	int num = 7;
	std::cout << " num in foo: " << num << std:endl;
	std::cout << "&num in foo: " << &num << std:endl;
	
	foo();
	
	return 0;
}

```
- After num has been allocated, foo is called. Foo's allocation of x has to happen below the allocation for  num in the main function. 
	- We expect num to have a higher memory address than x, because it was allocated first.

How can we use these memory addresses?

==Pointers==
- A variable that stores the memory address of the data.
	- Pointers are a level of indirection from the data.
	- We "follow" a pointer to where it's pointing to.
- In C++, a pointer is defined by adding an * to the type of the variable.
	- Integer pointer: `int * p = &num`
		- The type is int *
		- The variable name is p
		- And the value is going to be the memory address of our num variable.
	- So now we have an integer pointer that stores the value of where num is stored in memory.


Oftentimes we'll have a pointer, and we don't want to know what the contents of the *pointer* are (the memory address), instead we want to know the contents of the memory space that the point is pointing to.

There's a ==Dereference operator== where, given a pointer, a level of indirection can be removed with the dereference operator *
```cpp
int num = 7;
int * p = &num  // A pointer int pointing to the memory address of hte num variable
int value_in_num = *p  // Because we're using it after variable definition, we look at what p is pointing do, and copy that value (7) into our value_of_num variable.

*p = 42; // We say that the depreferenced value of p should be made equal to 42. This means that we're updating num's value, essentially.
```
- Teh final state of this program
	- num = 42
	- p points to the memory address of num
	- value_in_num will have the value of 7


```cpp
#include "Cube.h"
using uiuc::Cube;

Cube *CreateUnitCube() {
	Cube cube; // Create a cube
	cube.setLength(15);  // Update it 
	return &cube;  // Return the memory address of the cube
}

int main() {
	Cube *c = CreateUnitCube(); // Create a Cube pointer c in main's stack memory. The type of the memory is a Cube *, the name is c, and it takes on the value of CrateUnitCube(), which returns a pointer...
	someOtherFunction();
	double a = c->getSurfaceArea();
	double v = c->getVolume();
	return 0;
}
```
![[Pasted image 20231216000940.png]]
- After the execution of the first line of the main function, we've got two sort of.... areas of memory, where we have each stack frame. We've ot a pointer from main's memory into the CreateUnitCube's memory region, where the actual Cube lives.
- Recall that the stack frame exists only as lone as the functoin is running
	- Because we've returned from the fucntion CreateUnitCube...
![[Pasted image 20231216001422.png]]
- That memory is (likely not) still going to contain that actual cube information!
- So when we run someOtherFunction later, something crazy is going to happen to our memory!
![[Pasted image 20231216001516.png]]

When we actually run our code, what happens?
- Our compiler is immediately going to warn us:
	- warning: address of stack memory associated with local variable 'cube'  returned....
- When we actually run that code... after having added a Cout:
	- In his example, this actually resulted in the surface area and volume both outputting as 0.
	- This is not what we'd expect, but it's the result of the memory being reused by the system for another purpose (and it happened to fill up that memory with zeroes). That memory was deallocated and overwritten (later) when the CreateUnitCube() function exited, so our pointer in our main function pointed to a memory address that didn't contain what it used to contain.

Depending on the compiler, the above example may simply crash with segfault instead of actually running. This can happen when any example tries to demonstrate undefined behavior.
In  practice, you don't have to worry about complicated things like this if you just avoid undefined behavior.

Always be mindful of exactly what you're doing with memory!

Here's a program that does a little bit of everything:

```cpp
int main() {
	int num = 7; // Create a variable num with type of int and value of 7
	std::cout << "num" << num < std::endl;  // Output the value of the variable with the label num
	std::cout << "&num" << &num < std::endl; // Output hte memory address of the variable with the name num eg 0xfff...9

	int *p = &num;  // Create a new pointer that points to the address of num. The contents of p should be the value we printed out on the prev line.
	std::cout << "p" << p < std::endl;  // 0xfff...9
	std::cout << "&p" << &p < std::endl; // P's memory address should be slightly smaller than the value of p, because we allocated stack memory downards; let's say its 0xfff...4
	std::cout << "*p" << *p < std::endl;  // This is the value of the dereferenced value of p, which is that 7
	
	*p = 42;  // Change the dereferenced value of p to 42
	std::cout << "" << num < std::endl; // 42

	return 0;
}

```
![[Pasted image 20231216002542.png]]

Note that C++ compilers can transform and rearranged compiled code for optimization purposes -- in that case, the addresses of the variables in the stack layout may NOT be as predicted in the lecture; in practice, this won't affect your C++ programming for most applications, unless you're in very specific fields (eg security research)

Pointers are a key idea in C++ that unlock the power of what we do in C++.

----

Now let's talk about ==Heap Memory!==
- Heap Memory allows us to create memory completely independent of the lifecycle of a function! (unlike stack memory)
- If we need to create memory that's longer-lived than the lifecycle of a function, we HAVE to use heap memory!
- The only way to create heap memory in C++ is with the use of the `new` operator.
- The `new` operator returns a POINTER to the memory storing the (start of the) data -- and NOT an instance of the data itself!
- This means that the `new` c++ operator DOES THREE THINGS:
	- allocates memory on the Heap for some data structure
	- initializes that data structure 
	- returns a pointer to the start of that data structure
- The memory is ONLY RECLAIMED by the system when the pointer is passed to the `delete` operator!
	- This is extremely powerful, but also a chance to hurt ourselves!

Heap Memory
- The code below allocates two chunks of memory:
	- Memory to store an *integer pointer* on the STACK!  (This is for int * numPtr)
	- Memory to store an *integer* on the HEAP!  (This is for the value on the right side of the assignment operator)

```cpp
int * numPtr = new int;
```

![[Pasted image 20231216003550.png]]
Our stack memory points to our heap memory! This heap memory will exist for the entire length of our program, or until we use delete.

```cpp
int main() {
	int *numPtr = new int;  // Creates a pointer to anew int
	// This creates a variable numPtr on the stack memory that's a pointer to somewhere in heap memory, which stores an integer.

	// std::couts
	// *numPtr <- Dereferencing the pointer; We haven't put any memory into the heap, so we have no idea what's in this memory; it's whatever's laying around in the OS; some mysterious value
	// numPtr <- The value of the pointer is the memory address in teh heap.  Heap memory starts are low addresses and grows up, unlike stack memory, so this is going to be a LOW hexadecimal number
	// &numPtr <- Going to be a HIGH hexadecimal number (since this is the address of the pointer itself in the stack memory)
	
	*numPtr = 42; // Dereferences the pointer and sets the value at that address to 42
	// std::couts
	// *numPtr <- Dereerencing agian. The value is going to be 42
	// numPtr <- The address that numPtr points to in heap memory hasn't changed
	/// &numPtr <- The address of numPtr itself in stack memory hasn't changed

	return 0
}

```
- Above:
	- See that I mentioned that Heap memory starts at low addresses and grows up, in contrast to the stack memory addresses, which start at high addresses and grow down


Here's another:
```cpp
int main() {
	int *p = new int;  // Create a p pointer variable with in stack memory that points to a location in heap memory (that will be an int later, but for now just contains whatever was already in that memory)

	Cube *c = new Cube;  // Create a varible c in stack memory that's a pointer to a cube, and set the value to be an object 

	*p = 42;  // Set the dereferenced value of p to be 42

	(*c).setLength(4);  // Call the setLength method/function on the object (?) that's at the memory address that's c's value.

	delete c;  // We look at c, see what it's pointg to , and delete the memory that it's pointing to, giving it back to the system (the heap in this case)
	delete p; // We look at p, see what it's pointing to, and delete the memory, giving it back to the system (deleting the thing in th heap)

	// At this point we still have something left; we have the stack memory (that will be delete when main returns), but we also have some memory in that stack memory pointing to data that's been deleted!
	return 0;
}
```
- See this (*c)? You'll never see this in real code -- there's an ==ARROW OPERATOR== that lets us access the contents of a class when that class is being pointed to. More on that later.
- The solution to this problem is the null pointer

==nullptr==
- The C++ keyword `nullptr` is a pointer that point stot he memory address 0x0
	- It's defined as pointing to nowhere;
	- it is an address that is reserved and will never be used by the system 
	- always generate a SEGMENTATION FAULT if we ever access a null pointer
	- You can never delete 0x0

So what we really want to be doing is say:
```
delete c;
c = nullptr;
```
- After the line `delete c;`, the pointer c still stores the address of the deleted variable, which is no longer valid to later dereference, and is therefore dangerous.
- The pointer doesn't automatically get set to nullptr (which is 0x0), so we need to MANUALLY set c to nullptr if we want to better protect against coding mistakes!
- This is good practice whenever we use "delete" on a pointer; but in this particular example, the function is about to return is "main"; When the main returns, the program exits and the operating system will usually release all the allocated memory back to the system anyways. This isn't an excuse to be sloppy though!
	- You'll see later than many class objects have their "destructor" function that get called silently when the programs exist (and mistakes can trigger crashes even during that process) 


==Arrow Operator==
- When an object is stored via a pointer, access can be make made to member functions using the `->` operator
	- Allows us to dereference a pointer and get access to the member functions

```cpp
(*c).myMethod();
# is equivalent to
c->myMethod();

```

Let's look at some code:
```cpp
int main() {
	Cube *c1 = new Cube(); // Create a cube in heap memory, and a c1 variable in stack memory that points to a memory address in heap memory containing a cube
	Cube *c2 = c1  // Create c2 variable in stack memory that points to the value of c1, which is the address of the memory in heap memory that contains our cube. This means that both c1 and c2 are pointing to the same memory address in heap memory.

	c2->setLength(10);  // We use the arrow operator to acecss a member function on the dereferenced c2, and set he cube's length to 10

	delete c2;  // We delete the thing that c2 is pointing to (the cube in the heap memory)... We follow its pointer, delete the data ,and clear it out.
	delete c1; // We attempt to the delete the thing that c1 is pointing to, but it's already been deleted! IF we try and delte c1, our data isn't there -- we just deleted something that we shouldn't have deleted. The compiler should give us an error because we tried to double-free memory that doesn't exist.

	return 0;
}
```
- Note that the deletion of c1 above didn't "just silently work", it resulted in a compiler error!

-----

# Heap Memory Puzzles

Let's do some puzzles!
```cpp
int main() {
	// See that we can declare multiple variables in stack memory on a single line
	int i = 2, j = 4, k = 8;  

	// These could all be on one line, just like above
	// We declare pointers p, q, and r in stack memory that point to the memory addresses of our integers that are also in stack memory
	int *p = &i; // p is a pointer and points to i's memory
	int *q = &j; // q is a pointer and points to j's memory
	int *r = &k; // r is a pointer and points to k's memory
	
	// Recall that stack memory goes "high to low", so i will have a higher memory address than j, and k will have a higher memory address than p.
	// Print out i,j,k,derefececed p, dereferencedq, dereferecedr
	std::cout << i << j << k << *p << *q << *r << std::endl;  // Should be (2, 4, 8, 2, 4, 8)


	// Set the value of k (which is 8) to be the value of i (which is 2)
	k = i; 
	// Print out the values of i,j,k, as well as the dereferenced values of p, q, and r
	// Note that k's value should have changed (to be the value of i)
	std::cout << i << j << k << *p << *q << *r << std::endl;  // Should be (2, 4, 2, 2, 4, 2)

	// Set the value of p (currentl 0x9) to be the value of q (0x8)
	// p takes on the value of q; p no longer points to i; it points o the same place as q.
	p = q;
	// Note that p's value shouldhave changed be be = to q's value
	std::cout << i << j << k << *p << *q << *r << std::endl;  // Should be (2, 4, 2, 4, 4, 2)

	// Set the value at the derenced q pointer (4) to be the value at the dereferrenced r pointer (2).
	// The dereferenced value of q is j; the dereferenced value of r is k;
	// So j's value takes on k's value; so they're all 2's now.
	*q = *r;
	// See that j's value should have changed (to be the value of k)
	std::cout << i << j << k << *p << *q << *r << std::endl;  // Should be (2, 2, 2, 2, 2, 2)

	return 0;
}

```

![[Pasted image 20231216141629.png]]
After k=i (I think this is a good way to visualize this; See that the arrows are pointing to the boxes themselves (memory address), rather than to the values inside of them.)
![[Pasted image 20231216141921.png]]
After p=q
![[Pasted image 20231216142546.png]]
after *q = *r


Let's look at another one:

```cpp
int main() {
	
	// Declare a new integer pointer x in stack memory that points to an address in heap memory which is a new integer
	int *x  = new int;

	// I didn't know that we can do this, but I think this is a 
	// int &y is giving us a REFERENCE VARIABLE; we'll dive more into this next week. For now, all a refrence variable means is that it aliases another piece of memory, which allows us to give a name to a piece of memory...
	// y will alias the dereference value of x. You can think of this as saying that y is an alias for the dereferenced value of x.
	int &y = *x;
	
	y = 4;

	cout << &x << endl; // print the address of x  (it's the memory address on the stack; it's going to be a large number, because it's stack memory)
	cout << x << endl; // print the value of x  (it's the memory address on the heap; it's goin to be a small number, because it's heap memory)
	cout << *x << endl; // print the dereferenced value of x  (it's 4, because we changed the value of y, which is a reference variable for the dereferenced value of x.)

	cout << &y << endl; // print the address of y. This is going to be that memory address on the heap; some low number
	cout << y << endl; // This is going to be the value of y, which is 4.
	cout << *y << endl; // !!! This is... going to be the dereferenced value of y? But y isn't a pointer, it's a reference variable. THIS EXPLODES.


	return 0;
}

```
- We're going to see that we get a compiler error
	- indirection requires a pointer operator
	- We can't take an indirection of an integer itself; y is just an int. Because it's not a pointer, we can't dereference a non-pointer.


Another puzzle!

```cpp
int main() {
	int *p, *q; // Initialize some int variables in the stack memory; they're pointers, but don't point to anything yet!
	p = new int;  // Create a new int object (?) on the heap; I belive the new keyword returns the ADDRESS of the created object; set p, the pointer,'s value to this new address
	q = p;  // Set q's value to p's value (a memory address on the heap; some low number)

	*q = 8; // Set the value of the memory that q points to (the dereferenced value of pointer q) to be 8. Now both p and q point to some memory address in the heap memory that contains an 8 value.

	// We should see that these are both 8
	cout << *p << endl;
	cout << *q << endl;

	// q now takes on the value of the memory address of a new int object in the heap memory (a higher address than the first heap object, since heap memory grows from low values to high values, unlike stack memory)
	q = new int;
	*q = 9; // Set the dereferenced value of q to be 9;

	// We should see that *p is 8 and *q is 9
	cout << *p << endl;
	cout << *q << endl;

	return 0
}
```
![[Pasted image 20231216154154.png]]
After the q = new int; *q = 9;


Here's another puzzle!

```cpp
int main() {
	int *x; // Declare an integer pointer x in the stack memory. This is a high number because stakc memory grows "down" from high numnbers.
	int size = 3;  // Create a new object (?) in stack memory having a value of 3. Size is label (?) that points to the value of 3. If we wanted the memory address here we'd have to do &size.
		x = new int[size]; // This is new! We'er allocationg a sequence of integers, sequentially in memory, stored as an array. We use the new keyword, so this is going to be on the heap! 

	// We iterate through it in a standard c-style for loop; We populate the values in our array...
	for (int i = 0; i < size; i++) {
		x[i] = i + 3;
	}
	// The contents of our array are 3,4,5

	// Because we used the new keyword, we have to delete this memory from the heap as well.
	delete[] x;
}

```
- Notice that we had to do delete[] x rather than just delete x!
	- I asked ChatGPT why:
		- When you use new int[size], we're allocating memory for MULTIPLE objects (an array of integers). To correctly deallocate this memory, you need to use delete[] x.
			- delete is used for releasing memory for a single object. It calls the destructor for that object (fi it's a class type) and then frees the memory
			- delete[] is used for releasing memory allocated for arrays; it calls the destructor for each element in the array (if they're objects of a class with a destructor) and then frees the entire block of memory allocated for the array.


-----
## Reading: ==Headers== and ==Source Files== : Cpp code organization
- Because a C++ program's source coede exists among several files, we should udnerstand how the compiler pulls these pieces together:
	- .h (or .hpp) files are ==header files==; they usually have the 
		- declaration of objects
		- declarations of global functions
	- .cpp files are often called the ==implementation files==, or the ==source files== -- this is where the function definitions and main program logic go.

In general, the header files contain *declarations* (where classes and custom types are listed by name and type, and function prototypes give functions a type signature)
And the source files contain *iomplementation* (where function bodies are actually defined, and some constant values are initialized).


Let's look at the cpp-class example:

Cube.h header file
```cpp
#pragma once  
// all header files need to start with the above code. It's a preprocessor directive that ensures that a header file is only included once during the compilation process, regardless of how many times it's imported using #include. This helps prevent issues like double definitions/circular dependencies.


class Cube {
	public: // public members; public protection region
		double getVolume();
		double getSurfaceArea();
		void setLength(double length);

	private: // prviate members; private protection region
		double length_;

}

```
- The header file includes both the the pragma once preprocessor directive that stops the file from being included multiple times in a complex project, as well as includes the declaration of the Cube class, listing its members (and signatures if functions), but not the implementation.

In the Cube.cpp source file:
```cpp
#include "Cube.h"
// We include the header file

double Cube::getVolume() {
	// We have access to cube's public/private member variables
	return length_ * length_ * length_;
}

...

double Cube::setLength(double legnth) {
	// We can take arguments and update our class state with them
	length_ = length;
}

```
- Note that the first thing we do is to include Cube.h to include all of the text from the Cube.h file that's in the same directory. Because it's specified in quotes, the compiler expects it to be in the same directory as the current file, Cube.cpp. There' a different way to refer to filenames using the <> that we'll see below
- Note that sometimes SHORT class function bodies will be directly defined in the file where they are declared inside the class decoration. The compiler handles that situation in a special way automatically so that it doesn't cause problems in the linking stage, described below.

In main.cpp:
```cpp
#include <iostream>  // Import 
#include "Cube.h"  // Import of a Cube header file in the same directory

int main() {
	Cube c;  // It's interesting that I don't have to refer to the module that I'm importing here, I can just use the Cube. That's beccause I think the text is literally included at the top of this file during compilation.

	c.setLength(3.48);
	double volume = c.getVolume();
	std::cout << "Volume: " << volume << std::endl;

	return 0;

}


```
- Here, we used two different \#include directives! 
	- We included a standard library header from the system directory -- this is shown by use of <>
		- When we write \#include \<iostream\>, the compiler looks for the iostream header file in a system-wide library path that's located outside of your current directory.
		- The \#incldue "Cube.h" is just like in our Cube.cpp file; We have to include the necessary headers in every cpp file where they are needed. 
			- but in this case, there's no need to write \#incldue "Cube.cpp", becaues the function definitions in the Cube.cpp file will be compiled separately and then ==LINKED== to the code from the main.cpp file... We don't need to know how this works, at the moment.

![[Pasted image 20231216161911.png]]

- The cube.cpp files and main.cpp files make requests to include various header files.
	- Sometimes the compiler might automatically skip some requests because of #pragma once to avoid including multiple times in the same file.
- The contents of the requests header files are temporarily copied into the cpp source code file where they are included. The cpp file with all its extra included content will then be compiled into something called an ==object file==.
	- These are files with a `.o` extension
- Each cpp file is separately compiled into an object file!
	- So Cube.cpp will be compiled into Cube.o, and main.cpp will be compiled into main.o
- Although each cpp file needs to the appropriate headers included for compilation, this has to do with checking type information and declarations.
- The compiled object files themselves are allowed to rely on DEFINITIONS that appear in the other object files.
	- 💡❗==NOTE:== This is why it's okay that the main.cpp doesn't have the Cube function definitions included in it; as long as main.cpp does know about the type 
	- of information that the Cube function signatures define in Cube.h, the main.o file will be LINKED AGAINST the compiled definitions in the Cube.o file!
		- Sam translation: The use of the including of Cube.h (containing API contractS) in our main.cpp file is to make sure that we aren't using the objects/functions from the Cube.h file in any way in our main.cpp file that would result in (eg) a "TypeError" (waving hand here). So are you doing division on a string member variable? We'd like to catch that at compile time --- and that doesn't require actually pulling in the implementation code.
- The linker program will also link against system-wide object files, such as for iostream.
- After the compiler and linker programs finish processing your code, you will get an executable file as a result. In this case, that file is simply named ==main==!

Fortunately, we don't have to configure the compiler manually in this course -- We'll provide a Makefile to you for each project, which is kind of a script that tells the compiler how to build your program.

-----
## Reading: Compiling and Running a C++ Program

We include a ==Makefile== with each project in the course, which is a kind of script that tells the compiler how to build our program
- It gives instructions to the compiler and linker about WHICH source files to use.
- We just type `make` in the command line in the appropriate directory, and the makefile in that directory will be executed.
	- We can then run the generated `main` executable by typing `./main` in our command line.
		- If we didn't include the ./ (saying "In this directory"), our shell would be looking for a system-wide command with the name "main" instead.

If you want to clear out all of the compiled objects and executable files to ensure that your program gets recompiled FROM SCRATCH next time, you can type:
```shell
make clean
```
This seems to remove the generated main file

---------
## Reading: Useful Bash Terminal Commands

- sudo
	- The sudo prefix means a following command will be temporarily executed with system administrator permissions (the su comes from superuser/root)
- man
	- Get the manual page for other commands; man man gets the manual for the man command
- pwd
	- Print working directory
- ls
	- List directory contents
- cd
	- Change directory
- cp
	- Copy file
- mkdir
	- Make directory
- mv
	- move or rename file
- rm 
	- remove or delete a file



-----
## Reading: C++ Syntax Notes:  Basic Operators, If-Else, and Type Casting

Assignment vs Equality
- Note that the = is the assignment operator and == is the comparison operator that checks equality; don't mix these up or make a typo!
```cpp
#include <iostream>

int main() {
	int x = 2;
	int y = 2;

	if (x == y) {
		std::cout << "x and y are equal" << std:endl;
	}
	else {
		std::cout << "x and y are NOT equal" << std::endl;
	}

	reutrn 0;
}

```


If-Else
- The above-example also showed if/else; 
- You can chain these main times:

```cpp
if (conditioni1) {

} else if (condition2) {


} else if (condition3) {


} else {
	// If none of the other conditiosn are met
}

```

The else's make this mutually exclusive; only one of the cases will execute.

If you want to possible execute multiple cases, just do:

```cpp
if (x > 0) {
	...
}
if (x > 10) {
	...
}
if (x > 50) {

} else {
	// This exclusively applies to the latest if statement, in thsi case
}

```

There's a ==ternary operator== also called the ==conditional operator==, which is actually a combination of two operators: ==?== and ==:==
The basic format for the syntax is:

```
[Boolean-valued condition] ? [expression to evaluate to if true] : [expressoin to evaluate to if false]
```
You could express the above in an if/else situation too, but the ternary operator is evaluated at the level of an expression, so it can be a sub-expression within a larger expressoin!


So you can do things like:
```cpp

// This DOES work
int x = 5 < 10 ? 1 : 2;  // Evaluates to 1

// This DOES NOT WORK
int y = if (5 < 10) {1;} else {2;}
```
You can creatively use the ternary operator to make code more concise and compact, although some people find it confusing to read.
Probably don't go about nesting ternary operations, or use it more than is necessary.

Comparison Operators and Arithmetic Operators
There are a bunch of operators in C++ that are compatible with variosu types:
```cpp
#include <iostream>

int main() {
	if (3 <= 4) {
		std::cout << "3 is less than or equal to 4" << std::endl;
	}

	int x = 3;
	x += 2; // This increases x by two; it's the same thing as doing x = x + 2

	std::cout << "x should be 5 now: " << x << std::endl;

	return 0;
}


```


#### Type Casting
It's important to understand ==casting== in C++; This is a phenomenon where a value is AUTOMATICALLY converts to a different type in order to allow an operation to continue.
- If you try to add together a floating point `double` value an an integer `int` value, the int will be *converted* to double first, and the rseult of the addition will have type double as well!
- But if you say the double value to an int variable, the floating point value will be *truncated* to an integer!

```cpp
#include <iostream>

int main() {
	int x = 2;
	double y = 3.5;
	std::cout << "This reuslt wlil be a double with value 5.5: " << (x*y) << std::endl;

	int z = x + y; // This is an expression that's calculated as a double, but then it's cast BACK to an int!
	std::cout << "This result will be an int with teh value 5" << z << std::endl;]

	return 0;
}

```

We can even cast various values to the Boolean `bool` type!
- For numerics, nonzero numeric values will be considered TRUE, and only ZERO will be considered FALSE.
```cpp
#include <iostream>

int main() {
	if (0) {
		std::cout << "You won't see this text." << std::endl;
	}
	if (-1) {
		std::cout << "You WILL see this text!" << std::endl;
	}
	return 0;
}

```
You need to be aware that casts are happening invisibly in this code!

----
## Reading: C++ Syntax Notes: Block Scope, Loops

After viewing the lecture about stack memory, we should already have an intuitive idea about how ==block scope== works.
- The idea is that certain blocks of code, signified by the curly braces `{}`, create an inner stack ON TOP OF the previous stack, which can hide the pre-existing values. 
- The lifetime of stack variables inside a block is called the variable's ==**scope**==.
- Variables created on the stack inside the inner block are only in existence for the duration of the block. When the block ends, the inner stack is removed, and the pre-existing values in the outer scope are available again (because they're still ==in-scope==).

```cpp
#include <iosterma>

int main() {
	int x = 2;
	// Value of x is 2

	// Create an inner scope block within a function
	{
		int x = 3;
		int y = 4;
		// The inner scope value of x is 3
		// The inner scope value of y is 4
	}

	// Now that the inner block has closed, both the inner x and y are gone! The original x variable is still on the stack, and has its old value
	// x is 2 in this scope


	// We can't refer to y here, because it doesn't exist in this scope at all! Trying to access it would give us a compilation error.

	return 0;
}

```
- You can create an inner scope block anywhere in a function!

Some keywords can have a block of `{}` afterwards, which does create an inner block scope for the duration of the conditional block!
```cpp
#include <iostream>

int main() {
	int x = 2;
	// x is 2

	if (true) {
		int x = 3;
		// value of x is 3
	}

	// value of x is 2

	return 0;
}

```


#### Loops

For loops
- The for loop syntax lets you specify an iteration variable, a range for that variable, and an increment instruction. The syntax is:
```cpp

for (declaration; condition; incrementOperation) { loop body }

```
Be careful about whether you're redeclaring the iteration variable at block scope or not! here's an example:


```cpp
#include <iostream>

int main() {
	int x = -1;
	// Outside the loop, x is -1

	// A for loop where we redeclare x in the inner scope
	// SAM: It seems like the parens is included in the inner block scope...
	for (int x = 0; x <= 2; x++) {
		// In the loop, x will be 0, 1, 2, 3
	}

	/// The outer scope value of x here is still -1!

	// This version of a for loop is NOT redeclaring x, just inheriting access to the same x variable from the outer scope. This modifies the outer x directly!
	for (x = 0; x <= 2; x++) {
		//In the loop, x will be 0, 1, 2, 3, after which it won't drop into this curly code.
		// There's some nuance here that could totally be on the test, so look into it, Sam.
	}

	// Outside the loop, x is now 3! Because it was modified from the inner scope above [rather than being redeclared in teh scope]
	
}
```


While loops
```cpp
whlie(condition) { loop body}
```
a la
```cpp
#include <iostream>

int main() {
	int x = 0;
	std::cout << "This should show 0, 1, 2, 3:" << std::endl;;

	while (x <= 3) {
		std::cout << "x is now: " << x << std::endl; // 0, 1, 2, 3
		x++;  // Postfix increment oiperator; increase x by one
	}
	// x is now 4, since it wasn't redeclared above or anything
	return 0
}

```

There are also modern "range-based" loops:

```cpp
for (temporary variable declaration: container) { loop body}
```
We'll talk more about these later, though!


------


### Week 2 Quiz

```cpp
int *p;
p = new int;
*p = 0;
```
- The name of the variable is p, the type is a pointer to an integer, and the memory address of the variable is the value returned by the expression &p
- The value of the variable is NOT 0; it is the address of the integer object on the heap

The address of any memory location in the stack is larger than the address of any memory location in the heap.

```cpp
int *allocate_an_integer() {
	int *i = new int;
	*i = 0;
	return i;
}
```
This is a function that is intended to return a pointer to a location in memory holding an integer value initialized to zero.

```cpp
int *allocate_an_integer() {
	// This this is creating an integer object in the stack and returning its meemory address... But the object is in the stack, so it is deallocated when this function completes. So... yikes!
	int i = 0;
	return &i;
}

int main() {
	int *j; // Create an integer pointer in the stack
	j = allocate_an_integer(); // Get the memory address of some created integer
	int k = *j; // Set k to the value of dereferenced j pointer. I think this will blow up.
	return 0;
}

```
- Depending on the compiler settings, the compiler may report either a warning or a compilation error. If allowed to compile, this is value could be 0, or some other value, or the program may terminate due to a memoryfault.

```cpp
int i = 0;
int *j = &i;
```
This creates two allocations on the stack and none on the heap

```cpp
int *i = new int;
```
This creates one allocation on each the stack and heap

```cpp
int *i = new int;

*i = 0;

int &j = *i;

j++;
```
- Recall that & on the left side is a REFERENCE VARIABLE. It points in this case to the value dereferenced by i; think of it as a 
	- j is an alias for the dereferenced value of i
j++; increments the value pointed to by variable i by one


```cpp
int i = 0, j = 1;
int *ptr = &i;
i = 2;
*ptr = 3;
ptr = &j;
j = i;
*ptr = 4;
```
- How many different values get stored in the same address that variable i has during the execution of the code above?
	- 3

```cpp
class Pair {
	public: double a,b;
};

int main() {
	Pair *p = new Pair;
	p->a = 0.0;
	return 0;
}

```
- p->a is equivalent to which one of the following?
	- (*p).a


----------

## Week 2 Challenge






















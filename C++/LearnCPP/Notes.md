
A computer's CPU is incapable of speaking C++; the limited set of instructions that a CPU can understand directly is called ==machine code== (or an ==instruction set==).
- 10110000 01100001

Back when computers were first invented, programmers wrote programs directly in machine language, which is a very difficult and time-consuming thing to do.

Some CPUs process instructions that are always 32 bits long, whereas some other CPUs (like the x86/64 family from intel) have instructions that can be a variable length.

Each set of binary digits interpreted by the CPU into a command do some very specific job (like compare two numbers, or put this number in that memory location). But because different CPUs have different instruction sets, instructions that were written for one CPU type can't be used on a cPU that doesn't share the same instruction set -- so programs generally weren't **portable** to different types of systems.

High-level languages like C, C++, Pascal, and later others like Java, JavaScript, and Perl were developed so that the programmer could write programs without having to be as concerned about what kind of computer the program will run on.

A ==compiler== is a program that reads SOURCE code and produces a stand-alone executable program that can then be run. Once the code has been compiled into an executable, you don't need a compiler to run the program.
- Compilers have gotten very good at producing fast, optimized code.

High-level language source code is compiled by a compiler to produce an executable. This executable runs on hardware and produces program results.

An ==interpreter== is a program that directly executes the instructions in the source code without requiring them to be compiled into an executable first. These interpreters tend to be *more flexible* than compilers, but *less efficient*, because the interpreting process needs to be done every time the program is run, which means that the interpreter is needed every time the program is run.

High-level language source code is interpreted by the interpreter, then runs on hardware to produce program results.

----
Aside: Compilers vs Interpreters
- Compiler Advantages
	- Can see all the code up front and perform a number of analyses and optimizations on the code.
	- Can generate low-level code that performs the equivalent of high-level ideas like "dynamic dispatch" or "inheritance" in terms of memory lookups inside of tables... Therefore the resulting program needs to remember less information about the original code.
	- Compiled code is generally faster than interpreted code because the instructions executed are usually *just for the program itself*, whereas the interpreter option has instructions both from the program itself plus the overhead from the interpreter.
- Compiler Disadvantages:
	- Generally have a long "start-up" time, because of the cost of doing all the up-front analysis they do; In settings like web browsers where it's important to load code fast, compilers might be slower, because they optimize short code that won't be run many times.
- Interpreter advantages:
	- Tend to start up faster than compilers because they read the code and written and don't do expensive optimizations.
- Interpreter disadvantages:
	- Tend to have higher memory usage than compilers, because the interpreter needs to keep more information about the program available at runtime.
	- Some CPU time will be spent inside the code of the interpreter, which can slow down the program being run.

Many languages combine elements of both, to get the best possible combination of advantages; JAva's JVM is a good example!
- The Java code itself is compiled (into bytecode) and initially it's interpreted. The JVM can then find code that's run many, many times and compile it directly to machine code, meaning that "hot" code gets the benefits of compilation, while "code" code does not! Modern JS engines use similar tricks.

Note that languages aren't necessarily compiled *or* interpreted! C is usually compiled, but there are C interpreters available that make it easier to debug, for instance.

Languages like C, C, and Pascal are compiled, where as "scripting" languages like JavaScript and Perl tend to be interpreted.

----

High-level languages have many desirable properties:
1. Easier to read and write, because the commands are closer to natural languages than they are to machine code.
2. Require fewer instructions to perform the same task as in lower level languages, making programs more concise and easier to understand.
3. Can be compiled for many different systems, so you don't have to change the program that you write to run on different CPUs! This is called ==portability==.

Exceptions to portability:
	Many operating systems, like MS Windows, contain platform-specific capabilities that you can use in your code. These make it much easier to write a program for a specific operating system, but at the expense of portability.
	Some compilers also support compiler-specific extensions; If you use these, your programs won't be able to be compiled by other compilers that don't support the same extensions without modification.

-----------

### Introduction to C/C++

Before C++, there was C!
- The ==C language== was developed in 1972 by Dennis Ritchie at Bell Telephone laboratories, primarily as a systems programming language (a language to write operating systems with).
- Ritchie's primary goals were to produce a minimalistic language that was
	- easy to compile
	- allowed efficient access to memory
	- produced efficient code
	- was self-contained (not reliant on other programs)

C ended up being so efficient and flexible, that in 1973 Dennis Ritchie and Ken Thompson rewrote most of the UNIX operating system using C (rather than assembly).
- The portability afforded by doing this in C rather than in assembly meant that Unix could be run on different CPUs! This sped Unix's adoption, and C/Unix had their fortunes tied together, and C's popularity was in part tied to the success of Unix as an operating system.

C++ was later developed by Bjarne Stroustrup at Bell Labs as an extension to C, starting in 1979 -- it added many new features to the C language, and is perhaps best thought of as a superset of C, though this is not strictly true.
- C++'s claim to fame results primarily from the fact that it's an object-oriented language!
- Five major updates to the C++ language (C++11, C++14, C++17, C++20, C++23) have been made since then, each adding additional functionality.
	- C++11 in particular added a huge number of new capabilities, and is widely considered to be the new baseline version of the language.
	- Each new formal release of the language is called a *language standard* or *language specification*


The underlying philosophies of C and C++ can be summed up as "*trust the programmer*" which is both wonderful and dangerous. You won't be stopped from doing things that don't make sense. Knowing what you shouldn't do in C/C++ is almost as important as knowing what you *should* do.

C++ excels in situations where high performance and precise control over memory and other resources is needed! 
- Examples
	- Video games
	- Real-time systems (for transportation, manufacturing, etc...)
	- High-performance financial applications (e.g. high frequency trading)
	- Graphical applications and simulations
	- Productivity/office applications
	- Embedded software
	- Audio and Video Processing
	- Artificial intelligence and neural networks

-----

### Introduction to the compiler, linker, and libraries

Once we've written some `.cpp` source files, we want to use a C++ compiler to compile them into executable files! 

The C++ ==compiler== will sequentially go through each source file and do two things:
1. Checks the code to make sure that it follows the rules of the C++ language
	- If it doesn't, the compiler will give you some error to pinpoint what needs fixing, and the compilation process will be aborted until the error is fixed.
2. It translates your C++ source code into a machine language called an ==object file==. These object files are typically named *name.i* or *name.obj*, where *name* is the same name as the *name.cpp* source from that it was produced from.


After the compiler creates one or more object files, then *another* program (separate from the compiler) called the **==linker==** kicks in!
The job of the linker is threefold:
1. Takes all of the object files generated by the compiler and combines them into a single executable program!
2. In addition to be able to link object files, the linker is also capable of linking *library files* (which are collections of precompiled code "packaged up" for reuse in other programs). For example, C++'s standard library includes the *iostream* library, which creates functionality for printing text on a monitor, etc. It's very common for the standard library to get linked into your programs (along with any other downloaded libraries you might have used)
3. Makes sure that all cross-file dependencies are resolved properly. 
	- If you define something in on .cpp file, and then use it in another .cpp file, the linker connects the two together. If the linker is unable to connect a reference to something with its definition, you'll get a linker error, and the linking process will abort.
 
Once the linker is finished linking all the object files and libraries, you will have a single executable file that you can then run!

Because there are multiple steps involved, the term ==building== is often used to refer to the full process of converting source code files into a single executable that can be run. A specific executable produced as a result of the result of building is called a ==build==.
- Build automation tools (like `make`) are often used to help automate the process of building programs and running automated tests.

Now you can run your single executable and see whether it produces the output that you're expecting!














































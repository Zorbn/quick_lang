# (Quick) Programming Language
## Not named Quick, just a working title (because the original version was implemented in an hour or so).

### Goals:
- Uniformity
- Simplicity
- Performance
- Fun
    - It's a hobby language.

### TODOs
- Add a namespace system like C# where everything has a namespace depending on it's location, and you can use `using` to not have to write the full namespaced path.
    - The current hacky namespace system breaks down when you import a file, then call a function from that file that uses types from another file that you haven't imported in the first file...
      The current system basically pretends that anything you haven't imported doesn't exist which has tons of bugs associated with it, better to do things properly.
- This could probably help with methods too, ie: each struct/union/enum has it's own namespace that functions can live in to be methods, maybe....
- Once methods are added to structs, it might be better to call them objects instead? `obj Player {}` instead of `struct Player {}`.

#### Type Checking
- Check for duplicate names of variables, functions, types, fields, etc.

#### Project Structure
- Namespaces.
- Some form of incremental compilation.
- Maybe parallel compilation too?

#### Features
- Default parameters.
- Variadic arguments.
- `for elem in array {}`
- Possibly generic type inference, so specifiers aren't required?
- Top level const declarations.
- Proper escape sequences (right now they are cheated in by using C's behavior).
- Real string type, as opposed to String being a const char*.

#### Missing Things and Bug Fixes
- Allow generic specifiers on methods.
- Definition lookups for methods don't take into account the extending_type_kind_id so they may return the wrong method or not a method at all.
- Struct equality comparisons.
- Allow `sizeof` on complex types at compile time.
- Some form of debug info: https://learn.microsoft.com/en-us/cpp/preprocessor/hash-line-directive-c-cpp
- Stack traces.
- Make break only work on loops, not switch statements. If you break in a switch statment in a loop it should break out of the loop.
- Catch ambiguous symbol lookups.

#### Standard Library
- `func alloc<T>(value T) *T` alternative to malloc for simple situations.

#### Cleanup
- Code that generates and uses union `__WithTag/__CheckTag` needs factoring.
- Refer to TODOs in source files.

#### Notes
- Generics are handled like "templates", they get type checked once for each usage. This is nice for flexibility, eg. a generic function that contains `genericFoo.bar()` will only cause an error if that generic function is called with a type parameter that doesn't have a `.bar()` method. However, it means that a generic function has to get used for it to get type checked.
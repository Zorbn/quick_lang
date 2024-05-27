# (Quick) Programming Language
## Not named Quick, just a working title (because the original version was implemented in an hour or so).

### Goals:
- Uniformity
- Simplicity
- Performance
- Fun
    - It's a hobby language.

### TODOs

#### Type Checking
- Check for duplicate names of fields and environment variables.

#### Project Structure
- Some form of incremental compilation.
- Maybe parallel compilation too?

#### Features
- Default parameters.
- Possibly generic type inference, so specifiers aren't required?
- Conditional compilation (#if DEBUG, #if UNSAFE, etc)

#### Missing Things and Bug Fixes
- Allow `sizeof` on complex types at compile time.

#### Cleanup
- Refer to TODOs in source files.

#### Notes
- Generics are handled like "templates", they get type checked once for each usage. This is nice for flexibility, eg. a generic function that contains `genericFoo.bar()` will only cause an error if that generic function is called with a type parameter that doesn't have a `.bar()` method. However, it means that a generic function has to get used for it to get type checked.
- Structs can be allocated and freed with `new`/`delete`. A struct with a valid destructor (a method with the signature `Destroy(*val MyStruct) Void`) will have its destructor called automatically when it is `delete`'ed. Structs that are stack allocated with `scope` will have their destructor called at the end of the scope they were created in. If one of these keywords is not used, it is up to the developer to choose if/when to call the destructor.
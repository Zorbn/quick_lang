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
- D language creator talked about "poisoning" nodes that have errors, basically ignore errors from nodes that have children with errors to prevent error cascades. Maybe look into this?
- Code used by multiple files (eg. any of the files in Core) will have duplicate errors reported. Maybe instead of immediately reporting, they should be aggregated and printed after all typers have finished, that way duplicates can be removed before printing (eg. if one file has reported an error at a node/location, ignore any future errors for that node/location).

#### Cleanup
- Refer to TODOs in source files.

#### Notes
- Generics are handled like "templates", they get type checked once for each usage. This is nice for flexibility, eg. a generic function that contains `genericFoo.bar()` will only cause an error if that generic function is called with a type parameter that doesn't have a `.bar()` method. However, it means that a generic function has to get used for it to get type checked.
- Structs can be allocated and freed with `new`/`delete`. A struct with a valid destructor (a method with the signature `Destroy(*val MyStruct) Void`) will have its destructor called automatically when it is `delete`'ed. Structs that are stack allocated with `scope` will have their destructor called at the end of the scope they were created in. If one of these keywords is not used, it is up to the developer to choose if/when to call the destructor.
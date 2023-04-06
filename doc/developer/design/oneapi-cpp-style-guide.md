<!--
SPDX-FileCopyrightText: Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
-->

# [DRAFT] oneAPI С++ Style Guide for Performance Libraries
## Table of Content
1. [Preface](#preface)
2. [GN: General Naming][gn]
3. [GF: General Formatting][gf]
4. [FA: Files][fa]
5. [NS: Namespaces][ns]
6. [FU: Functions][fu]
7. [CS: Classes and Structures][cs]
8. [TP: Templates][tp]
9. [VC: Variables and Constants][vc]
10. [ST: Statements][st]
11. [MS: Macros][ms]
12. [DX: Doxygen Annotations][dx]

[gn]: #gn-general-naming
[gf]: #gf-general-formatting
[fa]: #fa-files
[ns]: #ns-namespaces
[fu]: #fu-functions
[cs]: #cs-classes-and-structures
[tp]: #tp-templates
[vc]: #vc-variables-and-constants
[st]: #st-statements
[ms]: #ms-macros
[dx]: #dx-doxygen-annotations

## Preface
### Applicability of the Style Guide
Individual libraries can make a decision to relax some constraints in internal
implementation details (_not exposed to the user_). The existing libraries left a
legacy of already written code following the other guidelines/practices. Being
consistent with the already written code may be important. For example, the
Google Style Guide says "If you find yourself modifying code that was written to
specifications other than those presented by this guide, you may have to diverge
from these rules in order to stay consistent with the local conventions in that
code [...] Remember that consistency includes local consistency, too".

However, you may notice some rules are marked with :small_orange_diamond: that
stands for _Mandatory_. These rules are mandatory to follow in public user
interface.

**Note:** This style guide will cover more than just naming/formatting aspects.
Despite you might violate some naming rules (e.g., use `snake_case` for
variables, etc.), it is highly discouraged to step aside from general practices
such as [ISOCPP guidelines][isocpp_guidelines].

### Compatibility with other style guides
Some libraries are restricted by third-party specifications or considered as a
part of third-party code base (for instance, LLVM Project) that defines their
own code style. In that case, it may be impossible to follow this Style Guide,
so we suggest to follow third-party style guide.

### Definitions
Term | Description
:---:|-------------
`PascalCase` | Naming convention in which the first letter of each word in a compound word is capitalized.
`camelCase` | Naming convention in which each word within a compound word is capitalized except for the first word.
`snake_case` | Naming convention in which all letters are in lower case and words are separated with underscore character (_).
Capitalized `SNAKE_CASE` | The same as `snake_case` but all letters are capitalized.
:small_orange_diamond: | Stands for _Mandatory_ in rules definitions. The rule is mandatory if it significantly affects look and fill of the public interface or public header files. Additionally, execution of mandatory rules shall be easily controlled automatically with clang-format or any linter.
:small_blue_diamond: | Stands for _Optional_ in rules definitions. The rule is optional if its impact to the public interface or public header files is not visible for the end user. Or it is hard to check automatically via clang-format or linters.

### C++ Version
Use at least C++14 in public headers, individual libraries can make decision to
use newer C++ standard, e.g., C++17. C++ version in the files not exposed to the
user is library-level decision.

## GN: General Naming
:small_orange_diamond: **GN1:** Use `snake_case` for all type names: classes,
structures, enums, type aliases.

:small_orange_diamond: **GN2:** Use `snake_case` for all **variables** (global,
local, files, function parameters), global and local **constants** (including
`constexpr`) and **functions** (member, non-member) and enum values.

:small_orange_diamond: **GN3:** Use capitalized `SNAKE_CASE` **only for
macros**.

:small_blue_diamond: **GN4:** Use `PascalCase` **for template
parameters**: type, non-type and template template parameters.

:small_blue_diamond: **GN5:** Prefer full names to abbreviations, within reason.

:small_blue_diamond: **GN6:** Do not abbreviate by deleting letters within a
word.

:small_blue_diamond: **GN7:** Give as descriptive a name as possible.

:small_blue_diamond: **GN8:** No magic numbers. Whenever you type a number in
your code, use a constant instead to document the number's meaning. Exceptions:
`0`, `1`, `-1`, `2`.

### Additional recommendation for public API
:small_blue_diamond: **GN8:** Do not use abbreviations that are ambiguous or
unfamiliar to readers outside your project.

:small_blue_diamond: **GN9:** Abbreviations that would be familiar to someone
outside your project with relevant domain knowledge are acceptable.

### Table of common terms and abbreviations
 Term or abbreviation | Description
----------------------|-------------
 `*_count` | Prefix for a variable representing numerical value
 `row_count` | Number of rows in matrix or tabular data structure.
 `column_count` | Number of columns in matrix or tabular data structure.
 `param/params` | Structure/class, variable that represents parameter of algorithm or operation.

### Examples
```cpp
int cache_size;      // No abbreviation
int num_components;  // "num" is a widespread convention
int num_features;    // "Feature" is a common machine learning concept
int conv_stride;     // "Conv" is a common deep learning abbreviation
int n, i;            // Can be used as a temporary variable, loop index
int nerr;            // Ambiguous abbreviation
int n_comp_conns;    // Ambiguous abbreviation
int pc_reader;       // Lots of things can be abbreviated "pc"
int cstmr_id;        // Deletes internal letters
```

## GF: General Formatting
:small_orange_diamond: **GF1:** Each line of text in the code shall be at most
120 characters long. **The recommended line length is 100.**

:small_orange_diamond: **GF2:** Use **only spaces** and 4 spaces for
indentation, never use tabs. The other number of spaces is acceptable when it is
reasonable for code alignment.

```cpp
std::int32_t foo(const parameter_1& p_1,
                 const parameter_2& p_2) { // <- 17 spaces is used to
                                           //    align function parameters
    // 4 spaces for function body indentation
    if (condition) {
        // 4 spaces for control flow statements
    }
}
```

```cpp
template <typename T_1,
          typename T_2, // <- 10 spaces to align
          typename T_3> //    template parameters
class bar : public base_1<T_1>,
            public base_2<T_2>,
            public base_3<T_3> { // <- 12 spaces for alignment
public:
    explicit bar(int); // <- 4 spaces for members indentation
};
```

:small_orange_diamond: **GF3:** The open curly brace is always on the end of the
last line of the statement (type, function, namespace declaration, control flow
statement or member initializer list) if **it fits on one line**. If the
statement does not fit on one line, the open curly brace may be moved to the
next line.
```cpp
int foo() { // <- curly brace here
    do_something();
}

if (condition) { // <- curly brace here
    do_something();
}
else { // <- curly brace here
    do_something();
}

if (condition) { // <- curly brace here
    do_something();
} else { // <- also possible
    do_something();
}

if (very_long_boolean_expression_1 &&
    very_long_boolean_expression_2)
{ // <- curly brace can be placed here as condition does
  //    not fit on one line and the author thinks it may improve
  //    readability
}

class tensor : public tensor_base {
public:
    explicit tensor(...)
        : tensor_base(...),
          member_1_(...), member_2_(...),
          member_3_(...), member_4_(...)
    { // <- curly brace can be placed here as member initializer list
      //    does not fit on one line and the author thinks it may improve
      //    readability
    }
};
```

:small_blue_diamond: **GF4:** There is never a space between the parentheses and
the parameters in function declaration/invocation or control flow statements.
```cpp
// Not recommended
int foo( int param_1, float param_2 );
if ( condition );
call_foo( value_1, value_2 );
for ( int i = 0; i < loop_count; i++ );

// Right
int foo(int param_1, float param_2);
if (condition);
call_foo(value_1, value_2);
for (int i = 0; i < loop_count; i++);
```

:small_blue_diamond: **GF5:** The close curly brace is on the last line by
itself and not on the same line as the  open curly brace.
```cpp
// Not recommended
if (condition) { do_something(); }

// Right
if (condition) {
    do_something();
}
```

:small_blue_diamond: **GF6:** Minimize use of vertical whitespace. In
particular, do not put more than **one or two** blank lines between functions or
class definitions.

## FA: Files
:small_orange_diamond: **FA1:** Filenames should be lowercase and can include
underscores `_`.

:small_orange_diamond: **FA2:** C++ header files exposable to the user shall end
in `.hpp`. In case the library has backward compatibility requirements, e.g.,
`.h` is used for historical reasons, maintaining backward compatibility has
higher priority.

:small_orange_diamond: **FA3:** C++ source files should end in `.cpp`.

:small_orange_diamond: **FA4:** All header files shall start with `#pragma once`
guards to prevent multiple inclusion, see section [Structure of
HeaderFiles][sohf] for more details. Library can make decision to use
`#ifdef`-based preprocessor guard if compilers in library's IP plan do not
support `#pragma once`. In this case, format of preprocessor guard is up to the
library.

:small_blue_diamond: **FA5:** If you have function declaration in header file
(e.g., `foo.hpp`), function definition shall be in the source file with the same
name (e.g., `foo.cpp`). We do not mark it as mandatory, since it is not always
possible.

:small_blue_diamond: **FA6:** Header files shall be self-contained. A
self-contained header file is one that does not depend on the context of where
it is included to work correctly. In other words, header files should include
all the definitions that they need to be fully compilable.

### Structure of Header Files
:small_blue_diamond: **FA7:** Each header file shall contain items in the
following order:
  - Copyright;
  - Single blank line;
  - Doxygen comments (optional);
  - Single blank line if Doxygen comments are present;
  - Preprocessor guard;
  - Single blank line;
  - Include statements (if there);
  - Single blank line if include statements are present;
  - Global macros* (if there);
  - Single blank line if macros statements are present;
  - Type/function declarations wrapped into namespaces;

**\* Note:** It is not necessary to put _all_ macro definitions here, sometimes
it is convenient to have macros closer to the place, where they are used. For
example, sometimes it makes much more sense to define macros inside functions
that use them (see [Macros][ms] for more details). However, if the macro is used
throughout the library, it is recommended to put it in header file between
includes and namespace declaration.

:small_blue_diamond: **FA8:** Each header file shall include other header files
in the following order:
  - C standard headers;
  - C++ standard headers;
  - Single blank line if C/C++ headers are present;
  - Third party libraries' header files (e.g., SYCL, TBB, OMP, etc.);
  - Single blank line if third party headers are present;
  - Project's header files;

:small_blue_diamond: **FA9:** The included header files shall be sorted
alphabetically within each group (C/C++ standard, third party, project's
headers).
```cpp
#include "project_name/file1.hpp"
#include "project_name/file2.hpp"
#include "project_name/file3.hpp"
#include "project_name/dir1/file1.hpp"
#include "project_name/dir2/dir1/file1.hpp"
#include "project_name/dir2/dir2/file2.hpp"
```

#### Complete Example
```cpp
/* Copyright */

/// @file
/// File description

#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "CL/sycl.hpp"

#include "project_name/dir1/file1.hpp"
#include "project_name/dir1/file2.hpp"
#include "project_name/dir2/file1.hpp"

#define PROJECT_NAME_DECLARE(a, b) /*...*/

namespace project_name {
namespace optional_namespace_1 {
namespace optional_namespace_2 {

/* Declarations/definitions */

} // namespace optional_namespace_2
} // namespace optional_namespace_1
} // namespace project_name

```

### Additional recommendations on Includes
If project introduces multiple public header files (e.g., Boost, TBB, oneDAL), the
public header files of the library shall be located in the
`include/<library_name>` directory of the release structure. The reason to do
this is to give a sense of specification when someone tries to use your library
and avoid conflicts with user's headers or headers from the other libraries.
Thus, to use a library, one has to use the code:
```cpp
// Use
#include "library_name/my_public_header.hpp"

// Instead of
#include "my_public_header.hpp"

// or
#include "project_name_my_public_header.hpp"
```

### Structure of Source Files
:small_blue_diamond: **FA10:** Each source file shall contain items in the
following order:
  - Copyright;
  - Include statements (if there);
  - Type/function declarations/definitions wrapped into namespaces;

:small_blue_diamond: **FA11:** Each source file shall include other header files
in the following order:
  - Corresponding header files containing declarations;
  - C standard headers;
  - C++ standard headers;
  - Third party libraries' header files (e.g., SYCL, TBB, OMP, etc.);
  - Project's header files;

#### Source File Template
```cpp
/* Copyright */

#include "project_name/folder/filename.hpp"

#include <tuple>
#include <type_traits>

#include <CL/sycl.hpp>

#include "onemkl/blas.hpp"
#include "project_name/dir3/file1.hpp"
#include "project_name/dir4/file1.hpp"

namespace project_name {
namespace optional_namespace_1 {
namespace optional_namespace_2 {

// Declarations/definitions

} // namespace optional_namespace_2
} // namespace optional_namespace_1
} // namespace project_name
```

## NS: Namespaces
:small_orange_diamond: **NS1:** Use `snake_case`: all lowercase, with
underscores (_) between words for all namespaces.

:small_orange_diamond: **NS2:** The name of a top-level namespace must be common
for the entire project and reflect the project name. A particular name of the
top-level namespace is a product-level decision.

:small_orange_diamond: **NS3:** No decision on `oneapi` namespace, so do not
introduce it until decision is made.

:small_orange_diamond: **NS4:** Do not indent content inside a namespace scope.
```cpp
namespace onedal {

   // Wrong! Do not indent
   class table { };

} // namespace onedal
```

:small_orange_diamond: **NS5:** Put each namespace on its own line when declaring
nested namespaces (see note for C++17 below).
```cpp
#include "onemkl/path_to_some_header.hpp"

namespace onemkl {
namespace lapack {

/* ... */

} // namespace lapack
} // namespace onemkl
```

**Note:** C++17 allows more compact syntax for namespace declaration, this
format is also allowed.
```cpp
namespace onemkl::lapack {
} // namespace onemkl::lapack
```

:small_blue_diamond: **NS6:** Use the namespace `detail` for all
functionality that exposed in public headers, but is not expected to be used by
the end user.

:small_blue_diamond: **NS7:** Place all code (with few exceptions) in a
namespace, avoid having definitions in a global scope.
```cpp
// Not recommended
cl::sycl::event onemkl_foo(const cl::sycl::buffer<float>& a);

// Right
namespace onemkl {
cl::sycl::event foo(const cl::sycl::buffer<float>& a);
} // namespace onemkl
```

It is acceptable to split definition of namespaces within the same file:
<table>
<tr>
<td>

```cpp
namespace onedal {

namespace interface1 {
/* ... */
} // namespace interface1

namespace interface2 {
/* ... */
} // namespace interface2

} // namespace onednnl
```

</td>
<td>

```cpp
namespace onedal {
namespace interface1 {
/* ... */
} // namespace interface1
} // namespace onednnl

namespace onedal {
namespace interface2 {
/* ... */
} // namespace interface2
} // namespace onednnl
```

</td>
</table>

**Exceptions:**
  - Includes. Do not wrap includes with namespace;
  - Macros. Macros can be either declared inside or outside namespace;
    ```cpp
    // Ok
    #define ONEMKL_GLOBAL_MACRO /* ... */

    namespace onemkl {

    // Also Ok
    #define ONEMKL_LOCAL_MACRO /* ... */
    // ...
    #undef ONEMKL_LOCAL_MACRO

    } // namespace onemkl
    ```
  - Forward declaration/definition of symbols from the other namespaces or C
    functions;

:small_blue_diamond: **NS12:** Namespace definition shall end with the following
comment: `}<one_whitespace>//<one_whitespace><namespace_name>`.

:small_blue_diamond: **NS8:** Do not introduce very deep nested namespaces in
public API, 1-2 namespaces including top-level namespace is optimal, 3 levels is
maximum.

:small_blue_diamond: **NS9:** Do not use **user-visible** `using namespace`
directive in **any header file**. It pollutes the namespace and destroys all
efforts to maintain namespaces. Despite this, `using namespace` directive is
allowed in function scope or source files.

**Unsafe**
```cpp
// file: public_header_file.hpp
namespace onedal {
using namespace onedal::data_management;
} // namespace onedal
```

**Safe**
```cpp
// file: public_header_file.hpp
namespace onedal {

void foo() {
    using namespace onedal::data_management;
}

} // namespace onedal
```

**Also acceptable**
```cpp
// file: source_file.cpp
using namespace onedal::data_management;
```

:small_blue_diamond: **NS10:** Do not use **user-visible**
namespace/function/class **aliases** in public header files, because anything
imported in a header file becomes a part of the public API. Despite aliases are
discouraged in public header files, **aliases which are invisible to user** can
be used, e.g., aliases in functions, private section of `class`/`struct` and
`detail` namespace.

**Note**: Sometimes it may be design decision to expose something to public API
via `using` directive. It is difficult to recognize intentional and not
intentional `using` automatically with the help of linters, so this rule is not
marked as mandatory. We highly encourage you to pay attention to such `using`
during code review.

**Note**: Aliases in public header files and the `detail` namespace (see
**NS6**) are formally user-visible, but we do not prohibit it.

**public_header_file.hpp:**
```cpp
// Wrong if it is not design decision
namespace onedal {
using table = onedal::data_management::table;
/* or */
using onedal::data_management::table;
} // namespace onedal

// Safe
namespace onedal {

void foo() {
    using table = onedal::data_management::table;
}

} // namespace onedal
```

**project_public_includes/internal/header_file.hpp:**
```cpp
namespace onednnl {
namespace internal {

// Ok, since it's internal namespace
using some_class_alias = SomeClass;

} // namespace internal
} // namespace onednnl
```

**source_file.cpp:**
```cpp
namespace onedal {
namespace svm {

// Ok, since it's .cpp file, but prohibited in public header files
namespace kf = onedal::kernel_function;

} // namespace svm
} // namespace onedal
```

## FU: Functions
:small_orange_diamond: **FU1:** Use `snake_case`: all lowercase, with
underscores between words for all function names.
```cpp
return_type class_name::function_name(type_1 param_name_1, type_2 param_name_2) {
    do_something();
}
```

:small_orange_diamond: **FU2:** Empty function body shall be at the same line as
function signature.
```cpp
// Wrong
void empty_foo(type arg) {
}

// Right
void empty_foo(type arg) {}
```

:small_blue_diamond: **FU3:** There is never a space between function name (or
operator) and open brace. This rule applies to both function
declaration/definition and call.

**Declaration**
```cpp
// Not recommended
void foo (type param_name);
void operator() (type param_name);
void operator bool ();

// Right
void foo(type param_name);
void operator()(type param_name);
void operator bool();
```

**Call**
```cpp
// Not recommended
const auto x = foo (param_1, param_2);

// Right
const auto x = foo(param_1, param_2);
```

:small_blue_diamond: **FU4:** Do not put function signature and body on the
same line. The only exception is empty body, in that case it is recommended to
put curly braces at the same line (see rule **FU4**).
```cpp
// Not recommended
std::int32_t get_something() const { return something_; }

// Right
std::int32_t get_something() const {
    return something_;
}
```

:small_blue_diamond: **FU5:** Function name should start with a verb.

:small_blue_diamond: **FU6:** Use pattern `get_<nouns>`/`set_<nouns>` for
getters and setters, the same name for both functions.
```cpp
std::int32_t get_component_count() const;
void set_component_count(std::int32_t value);
```

:small_blue_diamond: **FU7:** Wrap parameter lists which do not fit on a single
line like this:
```cpp
return_type class_name::function_name(type_1 param_name_1, type_2 param_name_2,
                                      type_3 param_name_3) {
    do_something();
}
```

This style is also possible:
```cpp
return_type class_name::function_name(type_1 param_name_1,
                                      type_2 param_name_2,
                                      type_3 param_name_3) {
    do_something();
}
```

This style of parameters wrapping is also allowed:
```cpp
return_type class_name::very_long_function_name(
        type_1 param_name_1, // <- 8 space indent
        type_2 param_name_2,
        type_3 param_name_3) {
    do_something();  // <- 4 space indent
}
```

Use the same wrapping scheme for lambda expressions:
```cpp
const auto lambda = [](type_1 param_name_1,
                       type_2 param_name_2,
                       type_3 param_name_3) -> return_type {
    do_something();
};
```

:small_blue_diamond: **FU8:** Pass function parameters of non-primitive type via
`const` reference. Do not use `const` only if the parameter is expected to be
modified in the function body or it is used as an output.
```cpp
return_type class_name::function_name(const type_1& param_name_1,
                                      const type_2& param_name_2) {
    do_something();
}
```

Sometimes it is necessary to pass copy of the object to the function, in that
case it is not required to mark it `const`.
```cpp
/* Allowed if the function requires copy of param_name_2 */
return_type class_name::function_name(const type_1& param_name_1,
                                      type_2 param_name_2) { // <- const is not required
    do_something();
}
```

## CS: Classes and Structures
:small_orange_diamond: **CS1:** Use `snake_case`: lower case and all words are
separated with underscore character (_).
```cpp
class numeric_table;
class image;
struct params;
```
:small_blue_diamond: **CS2:** Avoid Hungarian notation (e.g., suffix `_t`,
`_impl`, etc.) in user exposable classes.

:small_blue_diamond: **CS3:** Prefer the class keyword over `struct`. The
`struct` and class keywords behave almost identically in C++. We add our own
semantic meanings to each keyword, so you should use the appropriate keyword for
the data-type you are defining. Use the `struct` keyword only in the following
cases:
  - Type is a collection of fields that serves only to carry the data. This
    concept is [borrowed from Google C++ Style Guide][class_vs_struct]:
    > `struct` should be used for passive objects that carry data, and may have
    > associated constants, but lack any functionality other than
    > initializing/clearing the data members. All fields must be public, and
    > accessed directly rather than through getter/setter methods. The struct
    > must not have invariants that imply relationships between different
    > fields, since direct user access to those fields may break those
    > invariants. Methods should not provide behavior but should only be used to
    > set up the data members, e.g., constructor, destructor, `initialize()`,
    > `reset()`.

    _If more functionality or invariants are required, a `class` is more
    appropriate. If in doubt, make it a `class`._
    ```cpp
    struct params {
        float gamma;
        std::int64_t element_count;
    };
    ```

  - Type is stateless, e.g., type trait or functor that does not have other
    methods except `operator()`:

    **Functor example**
    ```cpp
    template <typename T>
    struct fma {
        T operator()(T x, T y, T z) const {
            return x * y + z;
        }
    };
    ```

    **Type trait example**
    ```cpp
    template <typename T>
    struct is_integer_type {
        static constexpr bool value = false;
    };

    template <>
    struct is_integer_type<std::int8_t> {
        static constexpr bool value = true;
    };

    template <>
    struct is_integer_type<std::int16_t> {
        static constexpr bool value = true;
    };

    /* ... */
    ```

### Constructors
:small_blue_diamond: **CS4:** The acceptable formats for initializer lists are
- When everything fits on one line:
  ```cpp
  my_class::my_class(int var) : some_var_(var) {
      do_something();
  }
  ```
- If the signature and initializer list are not all on one line, you must wrap
  before the colon, indent 8 spaces, put each member on its own line and align
  them:
  ```cpp
  my_class::my_class(int var)
          : some_var_(var),             // <- 8 space indent
            some_other_var_(var + 1) {  // lined up
      do_something();
  }
  ```

- As with any other code block, the close curly can be on the same line as the
  open curly, if it fits:
  ```cpp
  my_class::my_class(int var)
          : some_var_(var),
            another_var_(0) {}
  ```

:small_blue_diamond: **CS5:** [All rules for functions][fu] (except naming)
shall be applied to constructors.

:small_blue_diamond: **CS6:** Do not define implicit conversions. By default,
all single-parameter constructors (except copy and move constructors) and
conversion operators shall be preceded with the `explicit` keyword. Implicit
conversions can hide type-mismatch bugs, where the destination type does not
match the user's expectation.

Do not mark constructors or conversion operators explicit only if it required by
class design. Implicit conversions can sometimes be necessary and appropriate
for types that are designed to transparently wrap other types.

:small_blue_diamond: **CS7:** Avoid virtual method calls in constructors.

## TP: Templates
:small_blue_diamond: **TP1:** Use `PascalCase` **for template parameters**:
type, non-type and template template parameters.
```cpp
template <typename Type, std::int64_t Size>
class foo {};

template <template<typename> class TemplateType>
class bar {};
```

:small_blue_diamond: **TP2:** Prefer the `typename` keyword to the `class`.
**Note:** When specifying a template template, the `class` keyword must be used
until C++17. Prefer the `typename` for a template template in C++17 code.
```cpp
// Acceptable
template <class Type>
class foo {};

// Better
template <typename Type>
class foo {};

// Template template in C++17
template <template <typename> typename TemplateType>
class foo {};
```

## VC: Variables and Constants
:small_orange_diamond: **VC1:** Use `snake_case` for all variables, function's
parameters and constants.

:small_blue_diamond: **VC2:** Use variables and constant names followed by
**one** underscore (`_`) for private and protected class-level variables.

:small_blue_diamond: **VC3:** The assignment operator (`=`) shall be
surrounded by single whitespace.
```cpp
const auto val = get_some_value();
```

:small_blue_diamond: **VC4:** Avoid names which starts and ends with `_` in
function scope. These names are reserved for class-level variables.

:small_blue_diamond: **VC5:** Declare all variables as `const` by default and
omit it only if the variable is expected to be modified.

:small_blue_diamond: **VC6:** If a constant value is compile-time evaluated
prefer `constexpr` declaration over `const`.

## ST: Statements
### Conditionals
:small_orange_diamond: **ST1:** Each of the keywords
`if`/`else`/`do`/`while`/`for`/`switch` shall be followed by one space. Open
curly brace after condition shall be prepended with one space.
```cpp
while (condition) { // <- one space after `while` and one space before `{`
    do_something();
} // <- `;` is not required
```

:small_blue_diamond: **ST2:** Each of the keywords
`if`/`else`/`do`/`while`/`for`/`switch` shall always have accompany braces (even
if they contain single-line statement).
```cpp
// Not recommended
if (my_const == my_var)
    do_something();

// Right
if (my_const == my_var) {
    do_something();
}
```

:small_blue_diamond: **ST3:** The statements within parentheses for operators
`if`, `for`, `while` shall have no spaces adjacent to the left and right
parenthesis characters:
```cpp
// Not recommended
for ( int i = 0; i < loop_size; i++ ) ...;

// Right
for (int i = 0; i < loop_size; i++) ...;
```
:small_blue_diamond: **ST4:** No assignment shall occur within the condition of
`if`/`else` clause:
```cpp
if (pObj = malloc(n)) {
    do_something();
}
```

:small_blue_diamond: **ST5:** If the condition in `if`, `else`, `while` does not
fit on a single line, it shall be split into logical parts and written on
several lines (if it contains more than 1 logical operator):
```cpp
// If it fits on one line
if (my_const_1 < my_var && flag) {
    do_something();
}
if (my_const_1 < my_var && my_const_2 > my_var && flag) {
    do_something();
}

// If it does not fit on one line
if (very_long_variable_name < other_long_variable &&
    another_very_long_condition) { // <- Aligned by the first condition
    do_something();
}
```

:small_blue_diamond: **ST6:** The constant shall be placed on the left side of
the comparison operator when evaluating the equality of a variable with a
constant in if statement.
```cpp
const int my_const = 123;
// Not recommended
if (my_var == my_const) {
    do_something();
}

if (my_var != my_const) {
    do_something();
}
// Right
if (my_const == my_var) {
    do_something();
}

if (my_const != my_var) {
    do_something();
}
```

:small_blue_diamond: **ST7:** The `?` and `:` characters in ternary operator
shall be surrounded by one space.

:small_blue_diamond: **ST8:** Expression with ternary operator shall be written
on separate lines if its length exceeds the number of acceptable characters in a
line:
```cpp
// If it fits one line
const float my_var = (t < barrier) ? val_1 : val_2;

// If it does not fit on one line
float my_var = (very_long_condition_or_combined_conditions)
    ? some_long_expression_1 // <- 4 spaces
    : other_expression;
```
### Loops
:small_blue_diamond: **ST9:** No assignment shall be made in for statement
except of a single loop counter initialization
```cpp
// Not recommended
float my_var;
for (int i = 0, my_var = 0.0f; i < loop_size; i++) {
    do_something();
}

// Right
float my_var = 0.0f;
for (int i = 0; i < loop_size; i++) {
    do_something();
}
```
:small_blue_diamond: **ST10:** The statements within parentheses for operators
`for` shall be split into logical parts and written on several lines, in case if
its length exceeds the number of acceptable characters in a line or condition
contains more than 1 logical operator:
```cpp
// If it does not fit on one line
for (int i = 0; i < loop_size && some_very_long_condition; i++) ...;

// Split it
for (int i = 0;
     i < loop_size &&
     some_very_long_condition;
     i++) ...;
```

:small_blue_diamond: **ST11:** Empty loop bodies shall use an empty pair of
braces and include a comment inside of braces explaining the goal of the loop.

### Switches
:small_blue_diamond: **ST12:** Within a switch statement each case shall be
indented, the block of the code for each case shall be indented once more. The
switch statement shall always have a default case as the last switch-clause. The
default case shall contain either a statement or a comment. All cases including
the default case shall end with a break statement. Each case blocks in switch
statements can have curly braces or not, depending on your preference. If you do
include curly braces they shall be included for all blocks, even if they contain
single-line statement:
```cpp
switch (expression) {
    case label1:
        statement1;
        statement2;
        break;

    case label2:
        statement3;
        statement4;
        break;

    default:
        statement5;
        break;
}

// If you do include curly braces they shall be placed as shown below.
switch (expression) {
    case label1: {
        statement1;
        statement2;
        break;
    }
    case label2: {
        statement3;
        statement4;
        break;
    }
    default: {
        statement5;
        break;
    }
}
```

## MS: Macros
TBD

## DX: Doxygen Annotations
:small_blue_diamond: **DX1:** Use a block of at least two C++ comment lines for
the detailed descriptions, if the ones are applied:
```cpp
///
/// ... Detailed description ...
///
```
:small_blue_diamond: **DX2:** Use C++ style comment for the brief descriptions,
if the ones are applied:
```cpp
/// Brief description
```
:small_blue_diamond: **DX3:** To document the members of a file, struct, union,
class, or enum, it is sometimes desired to place the documentation block after
the member instead of before. For this purpose an additional `<` marker is used
in the comment block:
```cpp
int var; ///< Brief description of the member which ends at this dot. Details follow
         ///< here if the ones are required.
```

**Note**: Macro documenting should follows the rules of documenting of the method/function, variable, etc. in depend on macro definition, and documenting of the constant should follows the rule of documenting of the variable.

### Complete Example
```cpp
/// @file
/// Brief description of the file which ends at this dot. Details follow
/// here if the ones are required.

/// Brief description of the macro which ends at this dot. Details follow
/// here if the ones are required.
#define SOME_MACRO 1

/// Brief description of the global variable which ends at this dot. Details follow
/// here if the ones are required.
const int some_variable = 10;

/// Brief description of the enumeration which ends at this dot. Details follow
/// here if the ones are required.
enum some_enum {
    first_value = 1, ///< Brief description of the first value which ends at this dot. Details follow
                     ///< here if the ones are required.
    second_value = 2, ///< Brief description of the second value which ends at this dot. Details follow
                      ///< here if the ones are required.
    third_value = 3 ///< Brief description of the third value which ends at this dot. Details follow
                    ///< here if the ones are required.
};

/// Brief description of the class which ends at this dot. Details follow
/// here if the ones are required.
///
/// @tparam type_1  Type of some object
/// @tparam type_2  Type of some object
template <typename type_1, typename type_2>
class some_class {
public:
    /// Brief description of the method which ends at this dot. Details follow
    /// here if the ones are required.
    ///
    /// @param [in]   param_1  first value
    /// @param [out]  param_2  second value
    ///
    /// @return Description of the method result if it's required
    int some_method(type_1 param_1, type_2& param_2) {
        /* ... */
    }

public:
    /// Brief description of the variable which ends at this dot. Details follow
    /// here if the ones are required.
    int var3_;

    int var4_; ///< Brief description of the variable which ends at this dot. Details follow
               ///< here if the ones are required.
};
```

[isocpp_guidelines]: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
[sohf]: #structure-of-header-files
[class_vs_struct]: https://google.github.io/styleguide/cppguide.html#Structs_vs._Classes

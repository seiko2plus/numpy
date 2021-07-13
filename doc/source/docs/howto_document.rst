.. _howto-document:


A Guide to NumPy Documentation
==============================

.. _userdoc_guide:

User documentation
******************
- In general, we follow the
  `Google developer documentation style guide <https://developers.google.com/style>`_.

- NumPy style governs cases where:

      - Google has no guidance, or
      - We prefer not to use the Google style

  Our current rules:

      - We pluralize *index* as *indices* rather than
        `indexes <https://developers.google.com/style/word-list#letter-i>`_,
        following the precedent of :func:`numpy.indices`.

      - For consistency we also pluralize *matrix* as *matrices*.

- Grammatical issues inadequately addressed by the NumPy or Google rules are
  decided by the section on "Grammar and Usage" in the most recent edition of
  the `Chicago Manual of Style
  <https://en.wikipedia.org/wiki/The_Chicago_Manual_of_Style>`_.

- We welcome being
  `alerted <https://github.com/numpy/numpy/issues>`_ to cases
  we should add to the NumPy style rules.



.. _docstring_intro:

Docstrings
**********

When using `Sphinx <http://www.sphinx-doc.org/>`__ in combination with the
numpy conventions, you should use the ``numpydoc`` extension so that your
docstrings will be handled correctly. For example, Sphinx will extract the
``Parameters`` section from your docstring and convert it into a field
list.  Using ``numpydoc`` will also avoid the reStructuredText errors produced
by plain Sphinx when it encounters numpy docstring conventions like
section headers (e.g. ``-------------``) that sphinx does not expect to
find in docstrings.

Some features described in this document require a recent version of
``numpydoc``. For example, the **Yields** section was added in
``numpydoc`` 0.6.

It is available from:

* `numpydoc on PyPI <https://pypi.python.org/pypi/numpydoc>`_
* `numpydoc on GitHub <https://github.com/numpy/numpydoc/>`_

Note that for documentation within numpy, it is not necessary to do
``import numpy as np`` at the beginning of an example.  However, some
sub-modules, such as ``fft``, are not imported by default, and you have to
include them explicitly::

  import numpy.fft

after which you may use it::

  np.fft.fft2(...)

Please use the numpydoc `formatting standard`_ as shown in their example_

.. _`formatting standard`: https://numpydoc.readthedocs.io/en/latest/format.html
.. _example: https://numpydoc.readthedocs.io/en/latest/example.html

.. _doc_c_code:

Documenting C/C++ Code
**********************

Recently, NumPy headed out to use Doxygen_ along with Sphinx due to the urgent need
to generate C/C++ API reference documentation from comment blocks, especially with
the most recent expansion in the use of SIMD, and also due to the future reliance on C++.
Thanks to Breathe_, we were able to bind both worlds together at the lowest cost.

**It takes three steps to complete the documentation process**:

1. Writing the comment blocks
-----------------------------

Although there is still no commenting style set to follow, the Javadoc
is more preferable than the others due to the similarities with the current
existing non-indexed comment blocks.

.. note::
   If you have never used Doxygen_ before, then maybe you need to take a quick
   look at `"Documenting the code" <https://www.doxygen.nl/manual/docblocks.html>`_.

This is how Javadoc style looks like::

  /**
   * Your brief is here.
   */
  void function(void);

For line comment, insert a triple forward slash::

  /// Your brief is here.
  void function(void);

And for structured data members, just insert triple forward slash and less-than sign::

  /// Data type brief.
  typedef struct {
      int member_1; ///< Member 1 description.
      int member_2; ///< Member 2 description.
  } data_type;


Common Doxygen Tags:
++++++++++++++++++++

.. note::
   For more tags/commands, please take a look at https://www.doxygen.nl/manual/commands.html

``@brief``

Starts a paragraph that serves as a brief description. By default the first sentence
of the documentation block is automatically treated as a brief description, since
option `JAVADOC_AUTOBRIEF <https://www.doxygen.nl/manual/config.html#cfg_javadoc_autobrief>`_
is enabled within doxygen configurations.

``@details``

Just like ``@brief`` starts a brief description, ``@details`` starts the detailed description.
You can also start a new paragraph (blank line) then the ``@details`` command is not needed.

``@param``

Starts a parameter description for a function parameter with name <parameter-name>,
followed by a description of the parameter. The existence of the parameter is checked
and a warning is given if the documentation of this (or any other) parameter is missing
or not present in the function declaration or definition.

``@return``

Starts a return value description for a function.
Multiple adjacent ``@return`` commands will be joined into a single paragraph.
The ``@return`` description ends when a blank line or some other sectioning command is encountered.

``@code/@endcode``

Starts/Ends a block of code. A code block is treated differently from ordinary text.
It is interpreted as source code.

``@rst/@endrst``

Starts/Ends a block of reST markup. Take a look at the following example:

.. literalinclude:: examples/doxy_rst.h

.. doxygenfunction:: doxy_reST_example

2. Feeding Doxygen
------------------

Not all headers files are collected automatically. You have to add the desired
C/C++ header paths within the sub-config files of Doxygen.

Sub-config files have the unique name ``.doxyfile``, which you can usually find near
directories that contain documented headers. You need to create a new config file if
there's no one located in a path close(2-depth) to the headers you want to add.

Sub-config files can accept any of Doxygen_ `configuration options <https://www.doxygen.nl/manual/config.html>`_,
but you should be aware of not overriding or re-initializing any configuration option,
you must only count on the concatenation operator "+=". For example::

   # to specfiy certain headers
   INPUT += @CUR_DIR/header1.h \
            @CUR_DIR/header2.h
   # to add all headers in certain path
   INPUT += @CUR_DIR/to/headers
   # to define certain macros
   PREDEFINED += C_MACRO(X)=X
   # to enable certain branches
   PREDEFINED += NPY_HAVE_FEATURE \
                 NPY_HAVE_FEATURE2

.. note::
    @CUR_DIR is a template constant returns the current
    dir path of the sub-config file.

3. Inclusion directives
-----------------------

Breathe_ provides a wide range of custom directives to allow
including the generated documents by Doxygen_ into reST files.

.. note::
   For more information, please check out "`Directives & Config Variables <https://breathe.readthedocs.io/en/latest/directives.html>`_"

Common directives:
++++++++++++++++++

``doxygenfunction``

This directive generates the appropriate output for a single function.
The function name is required to be unique in the project.

.. code::

   .. doxygenfunction:: <function name>
       :outline:
       :no-link:

Checkout the `example <https://breathe.readthedocs.io/en/latest/function.html#function-example>`_
to see it in action.


``doxygenclass``

This directive generates the appropriate output for a single class.
It takes the standard project, path, outline and no-link options and
additionally the members, protected-members, private-members, undoc-members,
membergroups and members-only options:

.. code::

    .. doxygenclass:: <class name>
       :members: [...]
       :protected-members:
       :private-members:
       :undoc-members:
       :membergroups: ...
       :members-only:
       :outline:
       :no-link:

Checkout the `doxygenclass documentation <https://breathe.readthedocs.io/en/latest/class.html#class-example>_`
for more details and to see it in action.

``doxygennamespace``

This directive generates the appropriate output for the contents of a namespace.
It takes the standard project, path, outline and no-link options and additionally the content-only,
members, protected-members, private-members and undoc-members options.
To reference a nested namespace, the full namespaced path must be provided,
e.g. foo::bar for the bar namespace inside the foo namespace.

.. code::

    .. doxygennamespace:: <namespace>
       :content-only:
       :outline:
       :members:
       :protected-members:
       :private-members:
       :undoc-members:
       :no-link:

Checkout the `doxygennamespace documentation <https://breathe.readthedocs.io/en/latest/namespace.html#namespace-example>`_
for more details and to see it in action.

``doxygengroup``

This directive generates the appropriate output for the contents of a doxygen group.
A doxygen group can be declared with specific doxygen markup in the source comments
as covered in the doxygen `grouping documentation <https://www.doxygen.nl/manual/grouping.html>`_.

It takes the standard project, path, outline and no-link options and additionally the
content-only, members, protected-members, private-members and undoc-members options.

.. code::

    .. doxygengroup:: <group name>
       :content-only:
       :outline:
       :members:
       :protected-members:
       :private-members:
       :undoc-members:
       :no-link:
       :inner:

Checkout the `doxygengroup documentation <https://breathe.readthedocs.io/en/latest/group.html#group-example>`_
for more details and to see it in action.

.. _`Doxygen`: https://www.doxygen.nl/index.html
.. _`Breathe`: https://breathe.readthedocs.io/en/latest/


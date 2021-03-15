# Information Extraction
Developing an Information Extraction system, which fetches relational tuples from the unstructured documents on the Internet, based on a user seed query.

## Preprocessing
The following preprocessing and filtering steps were performed while extracting text from beautiful soup:
- Removed all the sections with tags: script, headers and noscript. Since, we are mostly interested in the body of the webpage (which contains majority of the content), we observed removing this sections reduced the number of sentences to parse, and improved the relations extracted.
- Removed leading and trailing whitespaces, alongwith any citations which might interfere with the model.
- Other preprocessing include, splitting multiple headlines into each line, and removing any extra new lines that might be present due to the structure of html.
- Finally, took only the first 20000 characters as suggested in the project description.

## References
- Preprocessing Inspiration: https://stackoverflow.com/a/24968429
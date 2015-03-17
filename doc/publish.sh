#!/bin/bash

# Generate new documentation and push to gitub pages using git-subtree technique to push the doc directory content to the gh-pages branch that github pages uses

# delete local gh-pages branch if it exists
git branch -D gh-pages

# create new branch out of contents in doc directory
git subtree split --prefix doc/html -b gh-pages

# push to github
git push -f origin gh-pages


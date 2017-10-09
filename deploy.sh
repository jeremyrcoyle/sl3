#!/bin/bash

set -o errexit -o nounset
PKG_REPO=$PWD
PKG_TARBALL=$(ls -1t *.tar.gz | head -n 1)
cd ..

addToDrat() {
  mkdir drat; cd drat

  ## Set up Repo parameters
  git init
  git config user.name "addToDrat"
  git config user.email "addToDrat@travis.ci"

  ## Get drat repo
  git remote add upstream \
    "https://$GH_TOKEN@github.com/jeremyrcoyle/sl3.git" 2>err.txt
  git fetch upstream gh-pages 2>err.txt
  git checkout gh-pages 2>err.txt
  git reset --hard 2>err.txt

  Rscript -e "drat::insertPackage('$PKG_REPO/$PKG_TARBALL', \
    repodir = '.', \
    commit='Travis publish sl3: build $TRAVIS_COMMIT', \
    fields='Commit')"
  git push --force upstream gh-pages 2>err.txt
}

addToDrat

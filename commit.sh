sudo rm -rdf data
rm -rdf *.user
rm -rdf *.pyc
rm -rdf __pycache__
git checkout master
git add --all
git commit -m "$1"
git push -u origin master


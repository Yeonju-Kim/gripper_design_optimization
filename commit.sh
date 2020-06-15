sudo rm -rdf data
rm -rdf *.user
rm -rdf *.pyc
rm -rdf *.txt
rm -rdf __pycache__
cd GraspMetric
sudo mv build dist pyGraspMetric.egg-info ../../
cd ..
git checkout master
git add --all
git commit -m "$1"
git push -u origin master
sudo mv ../build ../dist ../pyGraspMetric.egg-info ./GraspMetric

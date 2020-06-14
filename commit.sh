sudo rm -rdf data
rm -rdf *.user
rm -rdf *.pyc
rm -rdf __pycache__
sudo mv GraspMetric/build GraspMetric/dist GraspMetric/pyGraspMetric.egg-info ../
sudo mv ../build ../dist ../pyDiffNE.egg-info ./GraspMetric
git checkout master
git add --all
git commit -m "$1"
git push -u origin master


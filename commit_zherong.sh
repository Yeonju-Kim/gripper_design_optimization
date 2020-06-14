sudo rm -rdf data
rm -rdf *.user
rm -rdf *.pyc
rm -rdf __pycache__
cd GraspMetric
sudo mv build dist pyGraspMetric.egg-info ../../
cd ..
git checkout zherong
git add --all
git commit -m "$1"
git push -u origin zherong
sudo mv ../build ../dist ../pyDiffNE.egg-info ./GraspMetric


sudo rm -rdf data
rm -rdf *.user
rm -rdf *.pyc
rm -rdf __pycache__
cd GraspMetric
sudo mv build ../../
sudo mv dist ../../
sudo mv pyGraspMetric.egg-info ../../
cd ..
sudo mv ../build ../dist ../pyDiffNE.egg-info ./GraspMetric
git checkout zherong
git add --all
git commit -m "$1"
git push -u origin zherong


sudo rm -rdf data
rm -rdf *.user
rm -rdf *.pyc
rm -rdf *.txt
rm -rdf __pycache__
cd GraspMetric
rm -rdf *.user
sudo mv build dist pyGraspMetric.egg-info ../../
cd ..
sudo mv ../build ../dist ../pyGraspMetric.egg-info ./GraspMetric

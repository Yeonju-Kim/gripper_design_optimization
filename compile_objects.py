import lxml.etree as ET
import glob,ntpath,os

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def path_leaf_no_extension(path):
    name=path_leaf(path)
    if name.find('.')!=-1:
        return os.path.splitext(name)[0]
    else: return name

def auto_download():
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/ObjectNet3D_cads.zip'):
        os.system('wget -P data ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip')
    if not os.path.exists('data/ObjectNet3D'):
        os.system('unzip data/ObjectNet3D_cads.zip -d data')

def compile_objects(asset,object_file_name):
    names = []
    object_list = glob.glob(object_file_name)
    for obj in object_list:
        mesh=ET.SubElement(asset,'mesh')
        mesh.set('file',obj)
        names.append(path_leaf_no_extension(obj))
        mesh.set('name',names[-1])
    return asset,names

if __name__=='__main__':
    auto_download()
    asset=ET.Element('asset')
    asset,names=compile_objects(asset,'data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off')
    print(ET.tostring(asset,pretty_print=True).decode())
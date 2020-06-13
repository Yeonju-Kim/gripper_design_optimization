import lxml.etree as ET
import glob,ntpath,os
import trimesh as tm

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

def ensure_stl(obj,force=True):
    if (not obj.endswith('stl') and not obj.endswith('STL')) or force:
        ret=os.path.splitext(obj)[0]+'.stl'
        if not os.path.exists(ret) or force:
            mesh=tm.exchange.load.load(obj)
            mesh.export(ret)
        return ret
    else: return obj
    
def get_COM(obj):
    mesh=tm.exchange.load.load(obj)
    return mesh.center_mass,mesh.bounds

def compile_objects(asset,object_file_name,scale_obj=None,force=True):
    names = []
    fullnames = []
    object_list = glob.glob(object_file_name)
    for obj in object_list:
        obj=ensure_stl(obj,force)
        names.append(path_leaf_no_extension(obj))
        fullnames.append(os.path.abspath(obj))
        
        mesh=ET.SubElement(asset,'mesh')
        mesh.set('file',fullnames[-1])
        mesh.set('name',names[-1])
        if scale_obj is not None:
            mesh.set('scale',str(scale_obj)+' '+str(scale_obj)+' '+str(scale_obj))
    return names,fullnames

if __name__=='__main__':
    auto_download()
    asset=ET.Element('asset')
    compile_objects(asset,'data/ObjectNet3D/CAD/off/cup/[0-9][0-9].off')
    print(ET.tostring(asset,pretty_print=True).decode())
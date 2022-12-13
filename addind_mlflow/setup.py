# python setup.py build_ext clean
#cython: language_level=3


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import glob
import shutil
import os
import sys

###########################################
# build blacklist
build_blackList_file = ['run_main.py', 'run_st.py', 'setup.py']
build_blackList_folder = ['__pycache__', 'build', 'dist', 'hooks', '.vscode']


# files that dont copy to built version
copy_blackList_ext = [".py", '.md', '.txt', '.sh', 'xlsx']
copy_blackList_files = ['install.txt', 'run_st.spec']


create_main = True
###########################################

ext_modules = ""
cp = ""


def main():
    # delete old build
    try:
        shutil.rmtree('./build')
    except:
        pass

    FindBuildingFiles()

    try:
        BuildFiles()
        print('Build finished')
    except:
        CleanUpExcep()
        print("\nFailed to compile. Scroll up to see errors")
    else:
        # CleanUpExcep()
        CleanUp()
        CopyFiles()
        print("\nCompile finished!!!")


def FindBuildingFiles():
    print("Finding building files.......")
    print()
    # build ext modules
    global ext_modules, build_blackList_file

    ext_modules = []
    dirs = [x[0] for x in os.walk("./")]
    for d in dirs:
        for name in glob.glob(d + "/*.py"):
            name = name.replace("\\", "/")
            if isBlackList_Folder(name):
                continue
            filename = name.split('/')[-1:][0]
            if filename in build_blackList_file:
                continue
            p = name
            n = name[2:]
            n = n.replace("/", ".").replace("\\", ".").replace(".py", "")
            ext_modules.append(Extension(n, [p]))
            print(p)


def isBlackList_Folder(dir):
    global build_blackList_folder
    for b in build_blackList_folder:
        if "/" + b + "/" in dir:
            return True
    return False


def BuildFiles():
    print()
    print("Start building............")
    print()

    global ext_modules

    # start build
    try:
        setup(
            name='upload server',
            cmdclass={'build_ext': build_ext},
            ext_modules=cythonize(ext_modules, language_level="3")
        )
    except Exception as e:
        print(e)


def CleanUpExcep():
    print()
    print("Clean up and exist...........")
    print()

    global copy_blackList_ext
    global cp

    cp = []
    # copy non python files
    dirs = [x[0] for x in os.walk("./")]

    for d in dirs:
        for name in glob.glob(d + "/*.*"):
            name = name.replace("\\", "/")
            if isBlackList_Folder(name):
                continue
            isBlackListed = False
            for b in copy_blackList_ext:
                if name.endswith(".c"):
                    os.remove(name)
                    print(name)
                    isBlackListed = True
                    break
                if name.endswith(b):
                    isBlackListed = True
                    break


def CleanUp():
    print()
    print("Cleaning temp files...........")
    print()

    global copy_blackList_ext
    global cp

    cp = []
    # copy non python files
    dirs = [x[0] for x in os.walk("./")]
    com_dir = [x[0] for x in os.walk("./build")][1]
    for d in dirs:
        for name in glob.glob(d + "/*.*"):
            name = name.replace("\\", "/")
            if isBlackList_Folder(name):
                continue
            isBlackListed = False
            for b in copy_blackList_ext:
                if name.endswith(".c"):
                    os.remove(name)
                    print(name)
                    isBlackListed = True
                    break
                if name.endswith(b):
                    isBlackListed = True
                    break
            for b in copy_blackList_files:
                if name.endswith(f'/{b}'):
                    isBlackListed = True
                    break

            if not isBlackListed:
                cp.append(name)
                filename = com_dir + name[1:]
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except:
                        print('Failed to copy ' + name[1:])
                shutil.copyfile(name, filename)


def CopyFiles():
    print()
    print("Copying non python files...........")
    print()

    global cp

    for a in cp:
        print(a)

    # Create main function
    com_dir = [x[0] for x in os.walk("./build")][1]
    print(com_dir)

    if create_main:
        m = (
            'import app\n'
            'app.run()'
        )
        f = open(com_dir + "/run.py", "w")
        f.write(m)
        f.close()

        m = 'streamlit run run.py'
        f = open(com_dir + "/run.bat", "w")
        f.write(m)
        f.close()



if __name__ == "__main__":
    main()

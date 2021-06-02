# no need to use this script explicitly, just create a new release in github repo
rm -f build/*
rm -f dist/*
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine upload dist/*

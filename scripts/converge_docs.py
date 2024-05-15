import os

part  = 'serve'

root_dir = '/root/ray-repos/wzh-ray/doc/source/' + part
output_markdown_file = part + '-combined.md'

f = open(output_markdown_file, 'w')

for root, dirs, files in os.walk(root_dir):
    # print(root)
    for file in files:
        if file.endswith('.md'):
            file_path = os.path.join(root, file)
            # print(file_path)
            with open(file_path, 'r') as readfile:
                f.write(readfile.read() + '\n\n') 
f.close()
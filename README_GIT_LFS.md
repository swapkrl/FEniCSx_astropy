# Git LFS Setup for Large Files

This repository is configured to use Git Large File Storage (LFS) for files larger than 99MB.

## What is Git LFS?

Git LFS is an extension to Git that replaces large files with text pointers inside Git, while storing the file contents on a remote server. This helps keep your repository size small while still allowing you to work with large files.

## Setup

The repository is already configured with Git LFS. The following files have been set up:

1. `.gitattributes` - Defines which files should be tracked by Git LFS
2. `.git/hooks/pre-commit` - Automatically detects and tracks new large files with Git LFS
3. `setup_git_lfs.sh` - Script to set up Git LFS tracking for large files
4. `migrate_to_lfs.sh` - Script to migrate existing large files to Git LFS

## How to Use

### For New Files

When you add new files to the repository, the pre-commit hook will automatically detect files larger than 99MB and track them with Git LFS.

```bash
# Add files as normal
git add path/to/large/file.h5
git commit -m "Added large file"
git push
```

### For Existing Large Files

To migrate existing large files to Git LFS, run:

```bash
./migrate_to_lfs.sh
```

This will:
1. Find all files larger than 99MB
2. Ask for confirmation
3. Set up Git LFS tracking for these files
4. Re-add them to the Git index

After running the script, commit and push your changes:

```bash
git commit -m "Migrated large files to Git LFS"
git push
```

## Tracked File Types

The following file types are automatically tracked by Git LFS:

- `.bp` files (binary files used by DOLFINx)
- `.h5` files (HDF5 data files)
- `.xdmf` files (XDMF metadata files)
- All files in `outputs/data/vtx/` and `outputs/data/xdmf/` directories
- Any other file larger than 99MB

## Troubleshooting

If you encounter issues with Git LFS, try the following:

1. Make sure Git LFS is installed:
   ```bash
   git lfs install
   ```

2. Check if a file is tracked by Git LFS:
   ```bash
   git lfs ls-files | grep filename
   ```

3. If you need to track additional file types, edit the `.gitattributes` file and run:
   ```bash
   git add .gitattributes
   git commit -m "Updated Git LFS tracking"
   ```

4. If you're having issues pushing large files, make sure they're properly tracked by Git LFS:
   ```bash
   ./setup_git_lfs.sh
   ```

For more information, visit the [Git LFS documentation](https://git-lfs.github.com/).

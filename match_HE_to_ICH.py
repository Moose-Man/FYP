import os

# Define paths (train)
he_registered_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\train"
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"

# Define paths (test)
he_registered_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\test"
ihc_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\test"

# Get lists of image filenames (without extensions)
he_files = {f.split('.')[0] for f in os.listdir(he_registered_train_path) if f.endswith(".png")}
ihc_files = {f.split('.')[0] for f in os.listdir(ihc_train_path) if f.endswith(".png")}

he_test_files = {f.split('.')[0] for f in os.listdir(he_registered_test_path) if f.endswith(".png")}
ihc_test_files = {f.split('.')[0] for f in os.listdir(ihc_resized_test_path) if f.endswith(".png")}

# Find unmatched IHC images
extra_ihc_files = ihc_files - he_files  # Files in IHC that are NOT in HE

extra_ihc_test_files = ihc_test_files-he_test_files

# Delete unmatched IHC images
for extra_ihc in extra_ihc_files:
    extra_ihc_path = os.path.join(ihc_train_path, f"{extra_ihc}.png")
    if os.path.exists(extra_ihc_path):
        os.remove(extra_ihc_path)
        print(f"🗑️ Deleted extra IHC image: {extra_ihc_path}")

# Delete unmatched IHC images (test)
for extra_ihc in extra_ihc_test_files:
    extra_ihc_path = os.path.join(ihc_resized_test_path, f"{extra_ihc}.png")
    if os.path.exists(extra_ihc_path):
        os.remove(extra_ihc_path)
        print(f"🗑️ Deleted extra IHC image: {extra_ihc_path}")

# Final check
remaining_he = len([f for f in os.listdir(he_registered_train_path) if f.endswith(".png")])
remaining_ihc = len([f for f in os.listdir(ihc_train_path) if f.endswith(".png")])

print(f"✅ Final HE count: {remaining_he}")
print(f"✅ Final IHC count: {remaining_ihc}")

if remaining_he == remaining_ihc:
    print("🎯 Dataset is now perfectly matched! (train)")
else:
    print("⚠️ There may still be a mismatch—check manually! (train)")

# Final check
remaining_he_test = len([f for f in os.listdir(he_registered_test_path) if f.endswith(".png")])
remaining_ihc_test = len([f for f in os.listdir(ihc_resized_test_path) if f.endswith(".png")])

print(f"✅ Final HE count: {remaining_he_test}")
print(f"✅ Final IHC count: {remaining_ihc_test}")

if remaining_he == remaining_ihc:
    print("🎯 Dataset is now perfectly matched! (test)")
else:
    print("⚠️ There may still be a mismatch—check manually! (test)")

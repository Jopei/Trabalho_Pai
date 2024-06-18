import os
import cv2
import shutil
import imgaug.augmenters as iaa

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 25
dataset_size = 100

cap = cv2.VideoCapture(0)

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Define a sequência de data augmentation
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip 50% das imagens horizontalmente
    iaa.Affine(rotate=(-25, 25)),  # rotacionar entre -25 e 25 graus
    iaa.Affine(scale=(0.8, 1.2)),  # escalar entre 80% e 120%
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})  # transladar até 20% do eixo x e y
])

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Pronto? Pressione "Q" para iniciar, "R" para reiniciar a classe', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == ord('r'):
            clear_directory(class_dir)
            print('Diretório limpo. Pronto para coletar dados para a classe {}'.format(j))

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        # Data augmentation
        augmented_images = augmenters(images=[frame])

        for idx, aug_img in enumerate(augmented_images):
            cv2.imshow('frame', aug_img)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, '{}_{}.jpg'.format(counter, idx)), aug_img)

        counter += 1

cap.release()
cv2.destroyAllWindows()

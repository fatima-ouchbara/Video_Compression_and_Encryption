import re
import numpy as np
import shutil
from Crypto.Cipher import AES
import pywt
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity

class VideoProcessor:
    def __init__(self, input_video):
        self.input_video = input_video
        self.frames = []
        self.output_path = 'découpage_vidéo'



    def extract_frames_from_video(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        cap = cv2.VideoCapture(self.input_video)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            # Enregistrer chaque cinquième frame
            if count % 5 == 0:
                self.frames.append(frame)
                cv2.imwrite(os.path.join(self.output_path, f"frame{count}.png"),frame)

        cap.release()


        return self.frames, self.output_path

    def compress_frames_dwt(self, input_dir, output_dir, threshold):
        imvi="gray_frames"
        if os.path.exists(imvi):
            shutil.rmtree(imvi)
        os.makedirs(imvi)

        # Liste pour stocker les coefficients d'ondelettes compressés de chaque image
        compressed_coeffs_list = []
        count2 = 0
        # Boucle sur les fichiers dans le dossier d'entrée
        for filename in os.listdir(input_dir):
            # Chemin complet du fichier
            input_path = os.path.join(input_dir, filename)


            # Charger l'image en niveau de gris
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            count2 += 5
            output=os.path.join(imvi,f"compressed{count2}.png")
            cv2.imwrite(output, img)

            # Compresser les coefficients d'ondelettes de l'image
            coeffs = pywt.dwt2(img, 'haar')
            cA, (cH, cV, cD) = coeffs
            cA_compressed = pywt.threshold(cA, threshold, mode='soft')
            cH_compressed = pywt.threshold(cH, threshold, mode='soft')
            cV_compressed = pywt.threshold(cV, threshold, mode='soft')
            cD_compressed = pywt.threshold(cD, threshold, mode='soft')
            coeffs_compressed = cA_compressed, (cH_compressed, cV_compressed, cD_compressed)

            # Ajouter les coefficients compressés à la liste
            compressed_coeffs_list.append((coeffs_compressed, filename))

            # Convertir l'image compressée en uint8 pour l'enregistrement en niveau de gris
            img_compressed = pywt.idwt2(coeffs_compressed, 'haar')
            img_compressed = np.uint8(img_compressed)

            # Générer le nom de fichier de l'image compressée
            #output_file = os.path.join(output_dir, filename + "_compressed.jpg")
            output_file = os.path.join(output_dir, f"compressed{count2}.png")
            # Enregistrer l'image compressée en niveau de gris
            cv2.imwrite(output_file, img_compressed)

        return compressed_coeffs_list,output_file

    # Définir la clé de chiffrement AES
    global Block_size,key, iv, format
    Block_size = AES.block_size
    key = os.urandom(16)
    iv = os.urandom(Block_size)
    format = "PNG"


    def encrypt_image(self,input_file, output_dir):
        count = 0
        # Create an AES cipher object for encryption
        cfb_cipher = AES.new(key, AES.MODE_CFB, iv)
        for image in os.listdir(input_file):
            # Load the image
            img = cv2.imread(os.path.join(input_file, image))
            # Convert the image to bytes
            image_bytes = img.tobytes()

            # Encrypt the image bytes
            encrypted_bytes = cfb_cipher.encrypt(image_bytes)

            # Convert the encrypted bytes back to an image array
            encrypted_image_array = np.frombuffer(encrypted_bytes, dtype=np.uint8)
            count += 1
            # Reshape the image array to its original dimensions
            encrypted_image = encrypted_image_array.reshape(img.shape)
            output_file = os.path.join(output_dir, f"crypted_frame{count}.png")
            print("34",output_file)
            cv2.imwrite(output_file, encrypted_image)



    def decrypt_image(self,enc_file, output_dir):
        count = 0
        # Create an AES cipher object for decryption
        cfb_cipher = AES.new(key, AES.MODE_CFB, iv)
        for image_enc in os.listdir(enc_file):
            img = cv2.imread(os.path.join(enc_file, image_enc))
            # Convert the encrypted image to bytes
            encrypted_image_bytes = img.tobytes()

            # Decrypt the image bytes
            decrypted_bytes = cfb_cipher.decrypt(encrypted_image_bytes)

            # Convert the decrypted bytes back to an image array
            decrypted_image_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)

            # Reshape the image array to its original dimensions
            decrypted_image = decrypted_image_array.reshape(img.shape)
            count += 1
            output_file = os.path.join(output_dir, f"decrypted_frame{count}.png")
            cv2.imwrite(output_file, decrypted_image)

    def decompress_frames(self, frames_path,output_path,threshold):
        # Boucle sur les fichiers dans le dossier d'entrée
        for filename in os.listdir(frames_path):
            # Chemin complet du fichier
            frames = os.path.join(frames_path, filename)

            # Parcourir chaque fichier et décompresser l'image correspondante
        for frame in frames:
                # Charger l'image
                img = cv2.imread(os.path.join(frames_path, frame))

                # Appliquer une décomposition DWT 2D
                coeffs = pywt.dwt2(img, 'haar')


                # Extraire les coefficients LL, LH, HL et HH
                cA_compressed, (cH_compressed, cV_compressed, cD_compressed) = coeffs

                # décompression des coefficients d'ondelettes
                cA_decompressed = pywt.threshold_inverse(cA_compressed, threshold, mode='soft')
                cH_decompressed = pywt.threshold_inverse(cH_compressed, threshold, mode='soft')
                cV_decompressed = pywt.threshold_inverse(cV_compressed, threshold, mode='soft')
                cD_decompressed = pywt.threshold_inverse(cD_compressed, threshold, mode='soft')

                # reconstruction de l'image à partir des coefficients d'ondelettes décompressés
                coeffs_decompressed = cA_decompressed, (cH_decompressed, cV_decompressed, cD_decompressed)
                img_decompressed = pywt.idwt2(coeffs_decompressed, 'haar')

                # Appliquer une reconstruction inverse à partir des coefficients
                img_decompressed = pywt.idwt2((cA_decompressed, (cH_decompressed, cV_decompressed, cD_decompressed)), 'haar')

                # Convertir l'image de float en entier 8 bits
                img_decompressed = cv2.convertScaleAbs(img_decompressed)

                # Enregistrer l'image décompressée dans le dossier de sortie
                cv2.imwrite(os.path.join(output_path, frame), img_decompressed)

        # Retourner le chemin du dossier contenant les images décompressées
        return output_path

    def calculate_metrics(self, original_path, processed_path):
        """
        Paramètres :
        original_path (str) : Chemin d'accès au répertoire contenant les images originales.
        processed_path (str) : Chemin d'accès au répertoire contenant les images traitées.

    Retourne :
        dict : Un dictionnaire contenant le PSNR, le MSE et le SSIM pour chaque image.
        """
        # Get list of file names for each directory
        original_files = os.listdir(original_path)
        processed_files = os.listdir(   processed_path)

        # Initialize dictionary to store metrics
        metrics = {}

        # Loop over images in each directory and calculate metrics
        for i in range(len(original_files)):
            # Load images as numpy arrays
            original_image = cv2.imread(os.path.join(original_path, original_files[i]), cv2.IMREAD_GRAYSCALE)
            processed_image = cv2.imread(os.path.join(processed_path, processed_files[i]), cv2.IMREAD_GRAYSCALE)

            # Calculate metrics
            psnr = peak_signal_noise_ratio(original_image, processed_image)
            mse = mean_squared_error(original_image, processed_image)
            ssim = structural_similarity(original_image, processed_image)

            # Add metrics to dictionary
            metrics[original_files[i]] = {'PSNR': psnr, 'MSE': mse, 'SSIM': ssim}
            print(metrics)

        return metrics

    def measure_encryption_quality(self, original_frames_path, encrypted_frames_path):
        # Initialisation des variables de mesure
        histogram_diff = 0
        entropy_diff = 0
        corr_coef_diff = 0
        num_frames = 0

        # Récupération de tous les fichiers du dossier original
        original_files = os.listdir(original_frames_path)
        encrypted_files = os.listdir(encrypted_frames_path)

        for i in range(len(original_files)):
            # Chargement de l'image originale
            original_img = cv2.imread(os.path.join(original_frames_path, original_files[i]), cv2.IMREAD_GRAYSCALE)
            #original_img = cv2.imread(os.path.join(original_frames_path, file_name))

            # Chargement de l'image cryptée
            encrypted_img = cv2.imread(os.path.join(encrypted_frames_path, encrypted_files[i]), cv2.IMREAD_GRAYSCALE)

            # Calcul de l'histogramme et de la différence entre les deux histogrammes
            original_hist = cv2.calcHist([original_img], [0], None, [256], [0, 256])
            encrypted_hist = cv2.calcHist([encrypted_img], [0], None, [256], [0, 256])
            histogram_diff += cv2.compareHist(original_hist, encrypted_hist, cv2.HISTCMP_CORREL)

            # Calcul de l'entropie et de la différence entre les deux entropies
            original_entropy = self.calculate_entropy(original_img)
            encrypted_entropy = self.calculate_entropy(encrypted_img)
            entropy_diff += abs(original_entropy - encrypted_entropy)

            # Calcul du coefficient de corrélation et de la différence entre les deux coefficients
            corr_coef = np.corrcoef(original_img.ravel(), encrypted_img.ravel())[0, 1]
            corr_coef_diff += abs(corr_coef)

            num_frames += 1

        # Calcul de la moyenne de chaque mesure
        histogram_diff /= num_frames
        entropy_diff /= num_frames
        corr_coef_diff /= num_frames

        print('histogram_diff', histogram_diff, 'entropy_diff', entropy_diff, 'corr_coef_diff', corr_coef_diff)
        # Retour d'un dictionnaire avec les résultats de la mesure
        return {'histogram_diff': histogram_diff, 'entropy_diff': entropy_diff, 'corr_coef_diff': corr_coef_diff}

    def calculate_entropy(self, image):
        # Calcul de l'entropie d'une image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        hist_norm = hist_norm[np.nonzero(hist_norm)]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        print("@",entropy)
        return entropy

    def creer_video_images(self, input_folder_encrypt_frames, output_folder_encrypt_video):
        # Get a list of image files in the input folder
        pre_imgs = [f for f in os.listdir(input_folder_encrypt_frames) if
                    os.path.isfile(os.path.join(input_folder_encrypt_frames, f))]

        # Sort the files based on the numeric value in the file names
        file_list = sorted(pre_imgs, key=lambda x: int(re.findall(r'\d+', x)[0]))

        # Create a list of image file paths
        img = [os.path.join(input_folder_encrypt_frames, f) for f in file_list]

        # Check if there are any image files
        if len(img) == 0:
            print("No image files found in the input folder.")
            return

        # Read the first image to get the frame size
        frame = cv2.imread(img[0])
        if frame is None:
            print("Failed to read the first image.")
            return

        size = frame.shape[:2][::-1]  # Reverse the order of dimensions

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_full_path = os.path.join(output_folder_encrypt_video, "video_encrypt.mp4")
        video = cv2.VideoWriter(out_video_full_path, fourcc, 192, size)

        # Write each image to the video file
        for i, image_path in enumerate(img):
            print(image_path)
            frame = cv2.imread(image_path)
            if frame is not None:
                video.write(frame)
                print('Frame', i + 1, 'of', len(img))
            else:
                print('Failed to read image:', image_path)

        video.release()
        print('Output video saved to', out_video_full_path)

    def display_processed(self):
        # Display the processed video in a new window
        processed_video_path = r"C:\Users\DELL\Desktop\mini_projt\combine_frames_to_video_encrypt\video_encrypt.mp4"  # Replace with the actual path to your processed video
        cap = cv2.VideoCapture(processed_video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (650, 400))
            cv2.imshow('video decompressed and decrypted', frame)
            window_name = 'video decompressed and decrypted'
            cv2.moveWindow(window_name, 700, 190)  # Déplacer vers le coin supérieur gauche
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

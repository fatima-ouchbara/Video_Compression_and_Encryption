import os
import shutil
from customtkinter import filedialog
import customtkinter
from PIL import Image
import cv2
from code_application import VideoProcessor



class App:
    global threshold

    # creation des dossiers pour le stockage
    if os.path.exists("compressed_frames_dwt"):
        shutil.rmtree("compressed_frames_dwt")
    os.makedirs("compressed_frames_dwt")

    if os.path.exists("encrypted_frames"):
        shutil.rmtree("encrypted_frames")
    os.makedirs("encrypted_frames")
    if os.path.exists("decrypted_frames"):
        shutil.rmtree("decrypted_frames")
    os.makedirs("decrypted_frames")
    if os.path.exists("decompressed_frames_dwt"):
        shutil.rmtree("decompressed_frames_dwt")
    os.makedirs("decompressed_frames_dwt")

    def __init__(self, master):
        # creation d'interafce principale
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        self.master = master
        self.master.geometry("700x550")
        self.master.maxsize(700,550)
        root.iconbitmap("fileCrypt.ico")
        master.title("Application de Cryptage et Decryptage vidéo")
        # Frame principale
        self.frame1 = customtkinter.CTkFrame(self.master)
        self.frame1.pack(pady=50, padx=60, fill="both", expand=True)


        # image d'entrée
        current_dir = os.getcwd()
        image_path = os.path.join(current_dir, "lock.jpg")
        img_2 = customtkinter.CTkImage(Image.open(image_path), size=(600, 300))

        self.label_image = customtkinter.CTkLabel(self.frame1, text="", image=img_2)
        self.label_image.pack(pady=12, padx=10)

        self.btnComm = customtkinter.CTkButton(self.master, text="Commencer", command=self.commencer)
        self.btnComm.place(x=300, y=400)




    def commencer(self):
        # creation de frame ,label et un bouton pour selectionner la vidéo

        self.frame1.destroy()

        self.frame = customtkinter.CTkFrame(self.master)
        self.frame.pack(pady=50, padx=60, fill="both", expand=True)

        self.label1 = customtkinter.CTkLabel(self.frame, text="Sélectionnez une vidéo à traiter")
        self.label1.pack(pady=12, padx=10)

        current_dir = os.getcwd()
        image_path = os.path.join(current_dir, "ajouter.png")
        img_1 = customtkinter.CTkImage(Image.open(image_path), size=(20, 20))

        self.btnSelect = customtkinter.CTkButton(self.frame, image=img_1, text="Sélectionnez la vidéo",command=self.select_video)
        self.btnSelect.pack(pady=12, padx=10)

        # affichage de la vidéo
        self.labelVideoPath = customtkinter.CTkLabel(self.frame, text="")
        self.labelVideoPath.pack(pady=12, padx=10)

        self.btnHelp = customtkinter.CTkButton(self.master, text="A propos", command=self.aPropos)
        self.btnHelp.place(x=70, y=510)

        self.btnHelp = customtkinter.CTkButton(self.master, text="Quitter", command=self.master.destroy)
        self.btnHelp.place(x=490, y=510)

        self.labelVideo = customtkinter.CTkLabel(self.frame, text="")





    def select_video(self):
        global video_file
        video_file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Sélectionnez un fichier vidéo",
                                                filetypes=[
                                                    ("Video files", "*.mp4;*.avi;*.flv;*.mov;*.wmv;*.mkv;*.y4m")])

        if video_file:
            # create the video processor instance
            self.video_processor = VideoProcessor(video_file)
            self.playVideo = customtkinter.CTkButton(self.frame, text="Afficher la video", command=self.display_original)
            self.playVideo.pack(pady=12, padx=10)

            self.labelVideoPath.configure(text=video_file)


            self.frame2 = customtkinter.CTkFrame(self.frame)
            self.frame2.pack(pady=20, padx=10, fill="both", expand=True)
            # accéder au dossier courant
            current = os.getcwd()
            image = os.path.join(current, "lock_close.png")

            lock_close_2 = customtkinter.CTkImage(Image.open(image), size=(20, 20))
            self.btnCrypt = customtkinter.CTkButton(self.frame2, image=lock_close_2, text="  Cryptage vidéo",command=self.cryptage)
            self.btnCrypt.pack(pady=12, padx=10)

            image_5 = os.path.join(current, "lock_close.png")
            lock_open = customtkinter.CTkImage(Image.open(image_5), size=(20, 20))
            self.btnDecrypt = customtkinter.CTkButton(self.frame2,image=lock_open, text="Decryptage vidéo",command=self.decryptage)
            self.btnDecrypt.pack(pady=12, padx=10)

            self.label2 = customtkinter.CTkLabel(self.frame2, text="les boutons de test de Qualité")
            self.label2.pack(pady=12, padx=10)

            self.btnQCompress = customtkinter.CTkButton(self.frame2, text="Mesure de qualité",command=self.mesure)
            self.btnQCompress.pack(pady=12, padx=10)


    def display_original(self):

            original_video_path = video_file

            # Affichage de vidéo original

            cap = cv2.VideoCapture(original_video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (650, 400))

                cv2.imshow("Original Video", frame)
                # Déplacer la fenêtre vers le centre de l'écran
                window_name = "Original Video"
                cv2.moveWindow(window_name, 350, 190)  # Déplacer vers le coin supérieur gauche

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def aPropos(self):
        self.winApropos= customtkinter.CTk()
        self.winApropos.title("Fenêtre d'aide")
        self.winApropos.iconbitmap("fileCrypt.ico")
        self.winApropos.geometry("700x350")
        text_help = """« Application de Cryptage et de Décryptage vidéo » est une application de compression, cryptage ,
         décryptage et décompressions des vidéo, dans laquelle on a utilisé la compression basé sur la
         transformation en ondelette(DWT), et pour le cryptage on a choisi le chiffrement AES, pour utilisé 
         l’application vous cliquer sur « commencer » ,après vos cliquez sur « sélectionnez la vidéo » pour
         choisir la vidéo que vous souhaitez, dans cette étape la vidéo découper en plusieurs frames que vous 
         trouvez dans le dossier <découpage_vidéo>,puis vous cliquez sur cryptage vidéo pour compresser et crypté
         la vidéo, les résulta de cette partie sont enregistrer respectivement dans les dossier:
         <compressed_frames_dwt>,<encrypted_frames>, puis vous pouvez faire l’inverse à travers le clique sur le bouton
         "décryptage vidéo" qui permet de décrypter et decomprsser la vidéo, et bien sur vous pouvez
         mesurer la qualité de compression et de cryptage, et pour ça on a utilisé (MSE,PSNR et SSIM ) pour
         mesurer la  qualité de compression, et pour le cryptage on a évaluer leur qualité à partie d' 
         histogramme, L’entropie et les coefficient de corrélation"""
        self.labelHelp = customtkinter.CTkLabel(self.winApropos, text=text_help)
        self.labelHelp.pack(pady=12, padx=10)
        self.winApropos.mainloop()
    def cryptage(self):
        threshold = 5

        resultat_extract_frame,resultat_extract_folder = self.video_processor.extract_frames_from_video()
        self.video_processor.compress_frames_dwt(resultat_extract_folder, "compressed_frames_dwt", threshold)
        self.video_processor.encrypt_image("compressed_frames_dwt", "encrypted_frames")
        if os.path.exists("combine_frames_to_video_encrypt"):
            shutil.rmtree("combine_frames_to_video_encrypt")
        os.makedirs("combine_frames_to_video_encrypt")
        self.video_processor.creer_video_images("encrypted_frames","combine_frames_to_video_encrypt")
        self.video_processor.display_processed()

    def decryptage(self):
        threshold = 5
        self.video_processor.decrypt_image("encrypted_frames", "decrypted_frames")
        self.video_processor.decompress_frames("compressed_frames_dwt", "decompressed_frames_dwt", threshold)
    def mesure(self):
        self.video_processor.calculate_metrics("gray_frames", "compressed_frames_dwt")
        self.video_processor.measure_encryption_quality("compressed_frames_dwt", "encrypted_frames")

root = customtkinter.CTk()
app = App(root)
root.mainloop()

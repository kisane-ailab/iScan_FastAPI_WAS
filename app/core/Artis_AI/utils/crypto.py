from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

# AES 설정
KEY_LEN = 32  # 256 bits
IV_LEN = 16   # 128 bits

def encrypt_and_save(plaintext: str, filepath: str):
    key = os.urandom(KEY_LEN)
    iv = os.urandom(IV_LEN)

    # PKCS7 패딩
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    # 파일에 key + iv + ciphertext 저장
    with open(filepath, 'wb') as f:
        f.write(key)
        f.write(iv)
        f.write(ciphertext)

    print("🔐 암호화 및 저장 완료")
    return key, iv, ciphertext

def load_and_decrypt(filepath: str = "./utils/a31_train.bin") -> bytes:
    if not os.path.exists(filepath):
        print(f"{filepath} not exists!!")
        return b''

    with open(filepath, 'rb') as f:
        key = f.read(KEY_LEN)
        iv = f.read(IV_LEN)
        ciphertext = f.read()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # 패딩 제거
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

    return plaintext.decode('utf-8').encode('utf-8')

EVP_MAX_KEY_LENGTH = 64
EVP_MAX_IV_LENGTH = 16
def load_key_iv_and_encrypted_password(file_path):
    try:
        with open(file_path, 'rb') as f:
            key_full = f.read(64)  # 최대 키 길이 (C++과 동일)
            iv = f.read(16)        # IV 길이

            # 필요한 만큼만 자르기
            key = key_full[:32]    # AES-256 키는 32바이트

            encrypted_password = f.read()
        return key, iv, encrypted_password
    except Exception as e:
        raise RuntimeError("파일을 열 수 없습니다!") from e


def decrypt_password(encrypted_password, key, iv):
    # AES-256 uses 32-byte key and 16-byte IV
    key = key[:32]
    iv = iv[:16]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    padded_plaintext = decryptor.update(encrypted_password) + decryptor.finalize()

    # PKCS7 언패딩
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

    try:
        return plaintext.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("복호화된 데이터가 UTF-8 문자열이 아닙니다.")

def openssl_decrypt(file_path="./utils/Artis_Secure_Key.bin", debug_flag=False):
    decrypted_password = ""

    try:
        key, iv, encrypted_password = load_key_iv_and_encrypted_password(file_path)
        decrypted_password = decrypt_password(encrypted_password, key, iv)

        if debug_flag:
            print(f"[openssl_decrypt] 복호화된 ZIP 암호: {decrypted_password}")
    except Exception as e:
        print(f"[openssl_decrypt] 예외 발생: {e}")

    return decrypted_password.encode('utf-8')


if __name__ == '__main__':
    # 경로 설정
    '''BIN_FILE = './a31_train.bin'
    original = ""
    print("원문:", original)

    #암호화 및 저장
    encrypt_and_save(original, BIN_FILE)

    #복호화
    recovered = load_and_decrypt(BIN_FILE)
    print("복호문:", recovered)'''
    openssl_decrypt("./Artis_Secure_Key.bin")
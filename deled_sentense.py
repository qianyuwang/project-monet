
# img_mo = np.expand_dims(img_mo, axis=0)
# img_mo = np.append(img_mo,img_mo,axis=0)
# img_mo = Image.fromarray(img_mo).convert('RGB')








# img_mo.resize((int(output_width), int(output_height)),Image.BICUBIC)
# img_cl = img_cl.resize((int(output_width), int(output_height)),Image.BICUBIC)

img_mos = zip(img_mo1_1, img_mo1_2, img_mo1_3, img_mo2_1, img_mo2_2, img_mo2_3, img_mo3_1, img_mo3_2, img_mo3_3)
img_cls = zip(img_cl1_1, img_cl1_2, img_cl1_3, img_cl2_1, img_cl2_2, img_cl2_3, img_cl3_1, img_cl3_2, img_cl3_3)

img_mos['1_1'] = img_mo1_1 = ImageOps.crop(img_mo, (0, 0, w - pp, h - pp))
img_mos['2_1'] = img_mo2_1 = ImageOps.crop(img_mo, (pix, 0, w - pp - pix, h - pp))
img_mos['3_1'] = img_mo3_1 = ImageOps.crop(img_mo, (2 * pix, 0, w - pp - 2 * pix, h - pp))
img_mos['1_2'] = img_mo1_2 = ImageOps.crop(img_mo, (0, pix, w - pp, h - pp - pix))
img_mos['2_2'] = img_mo2_2 = ImageOps.crop(img_mo, (pix, pix, w - pp - pix, h - pp - pix))
img_mos['3_2'] = img_mo3_2 = ImageOps.crop(img_mo, (2 * pix, pix, w - pp - 2 * pix, h - pp - pix))
img_mos['1_3'] = img_mo1_3 = ImageOps.crop(img_mo, (0, 2 * pix, w - pp, h - pp - 2 * pix))
img_mos['2_3'] = img_mo2_3 = ImageOps.crop(img_mo, (pix, 2 * pix, w - pp - pix, h - pp - 2 * pix))
img_mos['3_3'] = img_mo3_3 = ImageOps.crop(img_mo, (2 * pix, 2 * pix, w - pp - 2 * pix, h - pp - 2 * pix))
img_mos['1_1'] = img_cl1_1 = ImageOps.crop(img_cl, (0, 0, w - pp, h - pp))
img_mos['2_1'] = img_cl2_1 = ImageOps.crop(img_cl, (pix, 0, w - pp - pix, h - pp))
img_mos['3_1'] = img_cl3_1 = ImageOps.crop(img_cl, (2 * pix, 0, w - pp - 2 * pix, h - pp))
img_mos['1_2'] = img_cl1_2 = ImageOps.crop(img_cl, (0, pix, w - pp, h - pp - pix))
img_mos['2_2'] = img_cl2_2 = ImageOps.crop(img_cl, (pix, pix, w - pp - pix, h - pp - pix))
img_mos['3_2'] = img_cl3_2 = ImageOps.crop(img_cl, (2 * pix, pix, w - pp - 2 * pix, h - pp - pix))
img_mos['1_3'] = img_cl1_3 = ImageOps.crop(img_cl, (0, 2 * pix, w - pp, h - pp - 2 * pix))
img_mos['2_3'] = img_cl2_3 = ImageOps.crop(img_cl, (pix, 2 * pix, w - pp - pix, h - pp - 2 * pix))
img_mos['3_3'] = img_cl3_3 = ImageOps.crop(img_cl, (2 * pix, 2 * pix, w - pp - 2 * pix, h - pp - 2 * pix))
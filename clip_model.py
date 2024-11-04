import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ExtendedModel(nn.Module):
    def __init__(self):
        super(ExtendedModel, self).__init__()
        self.base_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

        self.daily_text = ['a photo of Loungewear Style: Comfortable clothing intended for relaxing at home. Examples include sweatpants, hoodies, and pajamas.',
                            'a photo of Light Going Out Style: Casual yet presentable outfits suitable for quick errands or casual outings. Examples include jeans, t-shirts, and casual dresses.',
                            'a photo of Office Look: Professional attire appropriate for a workplace setting. Examples include dress shirts, blouses, slacks, skirts, and blazers',
                            'a photo of Formal Style: Dressy clothing for formal occasions, such as weddings or business events. Examples include suits, ties, evening gowns, and dress shoes.',
                            'a photo of Event Style: High-end attire for special events, similar to formal style but can include more elaborate designs. Examples include tuxedos, ball gowns, and cocktail dresses.',
                            'a photo of Sportswear Style: Athletic clothing designed for physical activities. Examples include tracksuits, athletic shorts, running shoes, and performance fabrics.']
        self.gender_text = ['a photo of Mannish Style: A style characterized by clothing and accessories traditionally associated with men, often incorporating tailored suits, button-down shirts, and other masculine elements, but worn by women or in a gender-neutral fashion',
                            'a photo of Unisex Style: Fashion that is designed to be worn by any gender, often featuring neutral or androgynous elements that do not emphasize traditional gender distinctions',
                            'a photo of Girlish Style: A style characterized by youthful, feminine clothing and accessories, often with playful, delicate, or romantic details',
                            'a photo of Elegant Style: A sophisticated and refined fashion style marked by graceful, polished, and often formal clothing that exudes class and understated luxury',
                            "a photo of Sexy Style: Fashion that emphasizes or accentuates the wearer's physical appeal, often through form-fitting, revealing, or provocative clothing and accessories"]
        self.embel_text = ['a photo of Style without decoration',
                            'a photo of Style with point decoration',
                            'a photo of Style with lots of decoration']
        
    def forward(self, x):
        input_daily = self.processor(text=self.daily_text,
                         images=x['ori_image'], return_tensors="pt", padding=True).to(DEVICE)
        input_gender = self.processor(text=self.gender_text,
                         images=x['ori_image'], return_tensors="pt", padding=True).to(DEVICE)
        input_embel = self.processor(text=self.embel_text,
                         images=x['ori_image'], return_tensors="pt", padding=True).to(DEVICE)
        
        output_daily = self.base_model(**input_daily)
        output_gender = self.base_model(**input_gender)
        output_embel = self.base_model(**input_embel)
        
        logits_daily = output_daily.logits_per_image  
        logits_gender = output_gender.logits_per_image  
        logits_embel = output_embel.logits_per_image 
        
        return logits_daily, logits_gender, logits_embel
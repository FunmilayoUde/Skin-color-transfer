from django.shortcuts import render
from django.core.files.base import ContentFile
from django.http import HttpResponse
from .forms import ImageUploadForm
from .skinsegmentation import load_skin_segmentation_model,create_skin_mask, refine_mask
from .colordetection import extractSkin,extractDominantColor, adjust_hsv_dominance
from .transfer import applyAdjustedColorToSkinRegion, blendSkinWithTexture


# Create your views here.


def home_page(request):
    return render(request, 'index.html')  # Assuming your template is named 'index.html'


def process_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Access uploaded images using form.cleaned_data['source_image'] and form.cleaned_data['target_image']
            source_image = form.cleaned_data['source_image']
            target_image = form.cleaned_data['target_image']

            # Perform your image processing using your existing scripts
            model = load_skin_segmentation_model('model_segmentation_realtime_skin_30.pth')
            skin_mask_source, resized_source = create_skin_mask(model,source_image)
            skin_mask_target, resized_target = create_skin_mask(model,target_image)

            refined_source_mask = refine_mask(resized_source, model, skin_mask_source)
            refined_target_mask = refine_mask(resized_target, model, skin_mask_target)

            source_skin = extractSkin(resized_source, refined_source_mask)
            target_skin = extractSkin(resized_target, refined_target_mask)

            dominant_color = extractDominantColor(source_skin, number_of_colors = 5, hasThresholding = True)
            adjusted_color = adjust_hsv_dominance(dominant_color, hsv_adjust = 2)

            
            target_skin_result = applyAdjustedColorToSkinRegion(target_skin, refined_target_mask, adjusted_color)
            # Generate the final image
            processed_image = blendSkinWithTexture(target_skin,refined_target_mask,target_skin_result, resized_target)
            
            processed_image_file = ContentFile(cv2.imencode('.jpg', processed_image)[1].tobytes())
            
            context = {
                'source_form': form,
                'result_image_file': processed_image_file,  # Pass the processed image file
            }
            return render(request, 'index.html', context)
    else:
        form = ImageUploadForm()
        context = {'form': form}
        return render(request, 'index.html', context)



function handleSourceImage(input) {
    if (input.files && input.files[0]) {
      const reader = new FileReader();
      reader.onload = function(e) {
        const image = new Image();
        image.onload = function() {
          // Update source image container with displayed image
          const container = document.getElementById('source-image-container');
          container.innerHTML = "";
          container.appendChild(image);
        };
        image.src = e.target.result;
      };
      reader.readAsDataURL(input.files[0]);
    }
  }
  
  function handleTargetImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
          const image = new Image();
          image.onload = function() {
            // Update target image container with displayed image
            const container = document.getElementById('target-image-container');
            container.innerHTML = "";
            container.appendChild(image);
          };
          image.src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
      }

    
  }
  
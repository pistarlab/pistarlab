<template>
<div v-if="item.ident">
    <div class="h-100 m-1">
        <div class="text-center image-box mb-3">
            <img :src="`${appConfig.API_URL}/api/env_preview_image/env_${item.ident}`" alt="No Image Found" style="max-height:200px;" />
        </div>
        <div class="h4 mb-1"> {{ item.displayedName }}</div>
        <div class="small " style="color:grey">{{ item.collection }}</div>
    </div>
</div>
</template>

<script>
import {
    appConfig
} from "../app.config";
export default {
    props: {
        item: Object
    },
    data() {
        return {
            appConfig

        };
    },
    mounted() {
        //
    },
    methods: {
        //

    },
    computed: {
        //

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style>

</style>

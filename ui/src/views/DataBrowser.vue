<template>
<div>

    <h3>
        <div>
            Current Directory:
            <span v-if="urlFilePath">{{ urlFilePath }}</span>
            <span v-else>Root</span>
        </div>
    </h3>
            <b-alert show variant="warning">Warning: This interface is buggy. If not working, close and try again.</b-alert>

    <b-link v-if="urlFilePath" @click="updateURL(parentLink)">Up</b-link>
    <div class="pt-2"></div>

    <b-table striped hover :items="itemList" :fields="fields" :dark="false" :small="true">
        <template v-slot:cell(name)="data">
            <b-link @click="updateURL(`${urlFilePath}${data.item.name}`)">{{ data.item.name }}</b-link>
        </template>
    </b-table>

</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    appConfig
} from "../app.config";

const fields = [{
        key: "name",
        label: "name",
        sortable: true
    },
    {
        key: "date",
        label: "Date Modified",
        sortable: true
    },
    {
        key: "is_dir",
        label: "is_directory",
        sortable: true
    },
    {
        key: "size",
        label: "size",
        sortable: true
    }
];

export default {
    name: "DataBrowser",
    components: {
        // SessionList
    },
    props: {
        path: String
    },
    data() {
        return {
            searchQuery: "",
            fields: fields,
            itemList: [],
            urlFilePath: "",
            urlParentPath: "",
            error: ""
        };
    },
    // watch: {
    //     // call again the method if the route changes
    //     $route: "fetchData"
    // },
    // Fetches posts when the component is created.
    created() {
        this.updateURL(this.path)
    },
    computed: {
        parentLink() {
            const parts = this.urlFilePath.split("/");

            const result = parts.slice(0, parts.length - 2).join("/");
            console.log("result: " + result);
            if (result == "") {
                return "";
            } else {
                return result;
            }
        }
    },
    methods: {
        updateURL(newUrlPath) {
            if (!newUrlPath || newUrlPath == "/") {
                newUrlPath = "";
            }
            this.urlFilePath = newUrlPath
            this.fetchData()

        },
        fetchData() {

            axios
                .get(`${appConfig.API_URL}/api/browser/` + this.urlFilePath)
                .then(response => {
                    if ("itemList" in response.data) {
                        this.itemList = response.data["itemList"];
                        this.urlFilePath = response.data["urlFilePath"];
                    } else if ("downloadURL" in response.data) {
                        window.open(`${appConfig.API_URL}` + response.data["downloadURL"]);
                    } else {
                        console.log("ERORR");
                    }
                })
                .catch(e => {
                    this.error = e;
                    console.log(this.error);
                });
        }
    }
};
</script>
